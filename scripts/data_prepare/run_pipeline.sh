# Config Path
RAW_DIR="raw_trajectory_path"
TRAIN_DIR="training_data_saving_dir"
TOKENIZER_PATH="huggingface_model_path"

# Script Dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPENAI_JSON_FILE="$TRAIN_DIR/openai_training_data.json"
OUTPUT_PARQUET="$TRAIN_DIR/train_$(date +%y%m%d)_128k.parquet"

echo "========================================="
echo "Start Data Pipeline"
echo "========================================="
echo "Original Data Dir: $RAW_DIR"
echo "Training Data Dir: $TRAIN_DIR"
echo "Tokenizer Path: $TOKENIZER_PATH"
echo "Output Parquet File: $OUTPUT_PARQUET"
echo

if [ ! -d "$RAW_DIR" ]; then
    echo "Error: No : $RAW_DIR"
    exit 1
fi

mkdir -p "$TRAIN_DIR"

echo "Step1: Transfer data to OpenAI..."
python3 "$SCRIPT_DIR/events2messages.py" --raw_dir "$RAW_DIR" --output_dir "$TRAIN_DIR"

if [ $? -ne 0 ]; then
    echo "Error: events2messages.py Execute Failed"
    exit 1
fi

if [ ! -f "$OPENAI_JSON_FILE" ]; then
    echo "Error: OpenAI JSON File not Generated."
    exit 1
fi

echo "✓ Step 1 Finished, Generate File: $OPENAI_JSON_FILE"
echo

echo "Step 2: Transfer to Parquet..."
python3 "$SCRIPT_DIR/messages_json2parquet_128k.py" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --input_json "$OPENAI_JSON_FILE" \
    --output_parquet "$OUTPUT_PARQUET" \
    --max_tokens 125000 \
    --duplicate_times 2

if [ $? -ne 0 ]; then
    echo "Error: messages_json2parquet_128k.py Failed"
    exit 1
fi

if [ ! -f "$OUTPUT_PARQUET" ]; then
    echo "Error: Parquet not generated"
    exit 1
fi

echo "✓ Step2 finished, Generate File: $OUTPUT_PARQUET"

file_size=$(du -h "$OUTPUT_PARQUET" | cut -f1)
echo

echo "========================================="
echo "Data Process Pipeline Finished!"
echo "========================================="
echo "Output Files:"
echo "  - OpenAI JSON: $OPENAI_JSON_FILE"
echo "  - Parquet File: $OUTPUT_PARQUET (Size: $file_size)"
echo