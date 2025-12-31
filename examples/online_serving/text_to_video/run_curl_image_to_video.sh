#!/bin/bash
# Wan2.2 image-to-video curl example

SERVER="${SERVER:-http://localhost:8094}"
INPUT_IMAGE="${1:-input.png}"
PROMPT="${2:-Make the scene come alive with gentle motion.}"
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT="${OUTPUT:-wan22_i2v_${CURRENT_TIME}.mp4}"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Input image not found: $INPUT_IMAGE"
    exit 1
fi

IMG_B64=$(base64 -w0 "$INPUT_IMAGE")

echo "Generating video..."
echo "Prompt: $PROMPT"
echo "Input: $INPUT_IMAGE"
echo "Output: $OUTPUT"

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"$PROMPT\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMG_B64\"}}
      ]
    }],
    \"extra_body\": {
      \"height\": 720,
      \"width\": 1280,
      \"num_frames\": 81,
      \"num_inference_steps\": 40,
      \"guidance_scale\": 4.0,
      \"seed\": 42,
      \"fps\": 24
    }
  }" | jq -r '.choices[0].message.content[0].video_url.url' \
  | sed 's/^data:video[^,]*,\s*//' | base64 -d > "$OUTPUT"

if [ -f "$OUTPUT" ]; then
    echo "Video saved to: $OUTPUT"
    echo "Size: $(du -h "$OUTPUT" | cut -f1)"
else
    echo "Failed to generate video"
    exit 1
fi
