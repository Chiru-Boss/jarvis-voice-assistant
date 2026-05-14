# NVIDIA Vision Integration

## Model

Default model: `llama-3.2-90b-vision-instruct`

## Environment

```env
VISION_ENABLED=true
VISION_MODEL=llama-3.2-90b-vision-instruct
NVIDIA_VISION_API_KEY=your_key_here
```

`core/vision_engine.py` sends screenshot + goal prompt and returns:

- `completed` (boolean)
- `analysis` (plain-English diagnosis)
- `recovery_plan` (list of suggested actions)
