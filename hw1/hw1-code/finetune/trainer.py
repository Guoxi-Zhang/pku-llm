import os
import math
import time
import inspect
import json

import torch
import transformers

from model import get_model_and_tokenizer
from dataset import get_dataloader
from utils import log, plot_learning_curve
from lora import get_lora_state_dict


class Trainer:

    def __init__(self, args):
        self.args = args
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.init_models()
        self.init_datasets()

        self.args.max_lr = self.args.lr
        self.args.min_lr = self.args.max_lr * 0.1
        self.args.total_train_steps = (self.args.epochs * len(self.train_dataloader)) // self.args.gradient_accumulation_steps
        self.args.total_eval_steps = len(self.eval_dataloader)
        self.args.num_warmup_steps = int(self.args.lr_warmup_ratio * self.args.total_train_steps)
        self.output_dir = os.path.join("./results", f"{self.args.output_dir_name}-{time.strftime('%Y%m%d-%H%M%S')}")
        assert not os.path.exists(self.output_dir), f"output directory {self.output_dir} already exists"
        os.makedirs(self.output_dir)
        
        # Save the args
        with open(os.path.join(self.output_dir, "arguments.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)

    def init_models(self):
        lora_args = None
        if self.args.use_lora:
            lora_args = {"part_module_name": self.args.lora_module_name, "lora_dim": self.args.lora_dim, "lora_scaling": self.args.lora_scaling, "lora_load_path": self.args.lora_load_path}
        self.model, self.tokenizer = get_model_and_tokenizer(self.args.model_name_or_path, self.args.trust_remote_code, self.args.max_length, self.args.use_lora, lora_args)
        self.model.to(self.device)
        self.optimizer = self.configure_optimizers()

    def configure_optimizers(self):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it <= self.args.num_warmup_steps:
            return self.args.max_lr * it / self.args.num_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.args.total_train_steps:
            return self.args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.args.num_warmup_steps) / (self.args.total_train_steps - self.args.num_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return self.args.min_lr + coeff * (self.args.max_lr - self.args.min_lr)

    def init_datasets(self):
        self.train_dataloader, self.eval_dataloader = get_dataloader(
            self.tokenizer, 
            self.args.data_path, 
            self.args.train_batch_size,
            self.args.eval_batch_size,
            self.args.eval_ratio,
            )

    def train(self):
        print('***** Running training *****')
        self.model.train()
        self.log_file = open(os.path.join(self.output_dir, "train_log.txt"), "w")
        step = 0
        self.optimizer.zero_grad()
        loss_accum = 0.0
        self.train_steps = []
        self.train_loss = []
        self.eval_steps = []
        self.eval_loss = []
        # Eval at the beginning
        e_t0 = time.time()
        e_loss = self.eval()
        e_t1 = time.time()
        self.eval_steps.append(0)
        self.eval_loss.append(e_loss)
        log(self.log_file, f"eval loss: {e_loss:.4f} | dt: {(e_t1 - e_t0)*1000: .2f}ms")
        t0 = time.time()
        for epoch in range(self.args.epochs):
            for batch in self.train_dataloader:
                step += 1
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / self.args.gradient_accumulation_steps
                loss_accum += loss.detach()
                loss.backward()
                if step % self.args.gradient_accumulation_steps == 0:
                    update_step = step // self.args.gradient_accumulation_steps
                    lr = self.get_lr(step / self.args.gradient_accumulation_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    t1 = time.time()
                    dt = t1 - t0
                    log_str = f"epoch: {epoch + 1}/{self.args.epochs} | step: {update_step:5d}/{self.args.total_train_steps} | loss: {loss_accum.item():.4f} | lr: {lr:.4e} | dt: {dt*1000: .2f}ms"
                    log(self.log_file, log_str)
                    self.train_steps.append(update_step)
                    self.train_loss.append(loss_accum.item())
                    loss_accum = 0.0
                    if update_step % self.args.eval_interval == 0:
                        e_t0 = time.time()
                        e_loss = self.eval()
                        e_t1 = time.time()
                        self.eval_steps.append(update_step)
                        self.eval_loss.append(e_loss)
                        log(self.log_file, f"eval loss: {e_loss:.4f} | dt: {(e_t1 - e_t0)*1000: .2f}ms")
                    t0 = time.time()   
        self.log_file.close()
        train_data = {
            "train_steps": self.train_steps,
            "train_loss": self.train_loss,
            "eval_steps": self.eval_steps,
            "eval_loss": self.eval_loss
        }
        with open(os.path.join(self.output_dir, "train_data.json"), "w") as f:
            json.dump(train_data, f, indent=4)
        plot_learning_curve(train_data, self.output_dir)
        self.save()

    def eval(self):
        self.model.eval()
        eval_loss = 0.0
        for batch in self.eval_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss / self.args.total_eval_steps
        self.model.train()
        return eval_loss.item()

    def save(self):
        print(f'Saving model to "{self.output_dir}" ...')
        if self.args.use_lora:
            lora_state_dict = get_lora_state_dict(self.model)
            torch.save(lora_state_dict, os.path.join(self.output_dir, "lora.pt"))
        else:
            self.model.config.to_json_file(os.path.join(self.output_dir, transformers.CONFIG_NAME))
            self.tokenizer.save_pretrained(self.output_dir)
            self.model.save_pretrained(self.output_dir)