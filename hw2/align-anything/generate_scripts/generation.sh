
if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

#=================== The template : need to change =========================================

MODEL=""
OUTPUT_DIR=""
OUTPUT_NAME=""
INPUTFILE=""
#=========================================================================
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL="$1"
			shift
			;;
		--model_name_or_path=*)
			MODEL="${arg#*=}"
			;;
        --output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
		    OUTPUT_DIR="${arg#*=}"
			;;
        --output_name)
			OUTPUT_NAME="$1"
			shift
			;;
		--output_name=*)
		    OUTPUT_NAME="${arg#*=}"
			;;
        --input_path)
			INPUTFILE="$1"
			shift
			;;
		--input_path=*)
		    INPUTFILE="${arg#*=}"
			;;
		--num_responses)
			NUM_RESPONSES="$1"
			shift
			;;
		--num_responses=*)
		    NUM_RESPONSES="${arg#*=}"
			;;

		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

python3  generation.py \
    --model_name_or_path ${MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --output_name ${OUTPUT_NAME} \
    --input_path ${INPUTFILE} \
	--num_responses ${NUM_RESPONSES}
