import requests

def unload_model(model_name, base_url="http://localhost:11434"):
    """
    Explicitly unloads a model from Ollama's memory by sending a request
    with keep_alive=0. This works via Ollama's native API.
    """
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "keep_alive": 0
            },
            timeout=30
        )
        response.raise_for_status()
        print(f"Unloaded model: {model_name}")
    except requests.RequestException as e:
        print(f"Failed to unload {model_name}: {e}")