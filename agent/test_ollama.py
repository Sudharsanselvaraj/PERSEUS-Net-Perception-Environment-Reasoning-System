@'
import httpx
r = httpx.post('http://localhost:11434/api/generate', json={
    'model': 'llama3.1:8b',
    'prompt': 'Respond with only a JSON object with these fields: action_type, message, tone, urgency, animation, reasoning',
    'stream': False,
    'format': 'json'
}, timeout=30)
print('Status:', r.status_code)
print('Response:', r.json().get('response',''))
'@ | Set-Content -Path test_ollama.py -Encoding UTF8