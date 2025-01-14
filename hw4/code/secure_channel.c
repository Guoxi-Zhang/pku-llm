/*
 * Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <string.h>

#include <doca_argp.h>
#include <doca_log.h>

#include <utils.h>

#include "secure_channel_core.h"

DOCA_LOG_REGISTER(SECURE_CHANNEL);

/*
 * Secure Channel application main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
	struct sc_config app_cfg = {0};
	struct cc_ctx ctx = {0};
	doca_error_t result;
	struct doca_log_backend *sdk_log;
	int exit_status = EXIT_SUCCESS;
	const char *json_file = NULL;

	// Parse command line arguments to get JSON file path
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
			json_file = argv[i + 1];
			break;
		}
	}

#ifdef DOCA_ARCH_DPU
	app_cfg.mode = SC_MODE_DPU;
#endif

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	/* Parse cmdline/json arguments */
	result = doca_argp_init("doca_secure_channel", &app_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}
	result = register_secure_channel_params(&app_cfg, json_file);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse register application params: %s", doca_error_get_descr(result));
		doca_argp_destroy();
		return EXIT_FAILURE;
	}
	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		doca_argp_destroy();
		return EXIT_FAILURE;
	}

	// // 测试无效参数
	// struct sc_config invalid_cfg = {0};
	// result = sc_start(&invalid_cfg, &ctx);
	// if (result != DOCA_SUCCESS) {
	// 	DOCA_LOG_ERR("Expected failure with invalid parameters");
	// 	// 模拟连接丢失
	// 	if (ctx.ep != NULL) {
	// 		DOCA_LOG_INFO("Simulating connection loss...");
	// 		doca_comm_channel_ep_disconnect(ctx.ep, ctx.peer);
	// 		sleep(1); // 等待断开连接
	// 		DOCA_LOG_INFO("Try to reconnect after connection loss");
	// 		result = sc_start(&app_cfg, &ctx); // 尝试重新连接
	// 		if (result != DOCA_SUCCESS) {
	// 			DOCA_LOG_ERR("Failed to reconnect after connection loss");
	// 		}
	// 	}
	// }
	
	/* Start Host/DPU endpoint logic */
	result = sc_start(&app_cfg, &ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to initialize endpoint: %s", doca_error_get_descr(result));
		// 模拟连接丢失
		if (ctx.ep != NULL) {
			DOCA_LOG_INFO("Simulating connection loss...");
			doca_comm_channel_ep_disconnect(ctx.ep, ctx.peer);
			sleep(1); // 等待断开连接
			DOCA_LOG_INFO("Try to reconnect after connection loss");
			result = sc_start(&app_cfg, &ctx); // 尝试重新连接
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Failed to reconnect after connection loss");
				exit_status = EXIT_FAILURE;

			}
		}
	}

	/* ARGP cleanup */
	doca_argp_destroy();

	return exit_status;
}
