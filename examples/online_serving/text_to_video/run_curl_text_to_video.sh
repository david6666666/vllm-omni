#!/bin/bash
# Wan2.2 text-to-video curl example

SERVER="${SERVER:-http://localhost:8093}"
PROMPT="${PROMPT:-A serene lakeside sunrise with mist over the water.}"
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT="${OUTPUT:-wan22_t2v_${CURRENT_TIME}.mp4}"

echo "Generating video..."
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ],
    \"extra_body\": {
      \"height\": 720,
      \"width\": 1280,
      \"num_frames\": 81,
      \"num_inference_steps\": 40,
      \"guidance_scale\": 4.0,
      \"guidance_scale_2\": 4.0,
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
