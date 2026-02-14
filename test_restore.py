import requests

file_path = 'velovision_backup_8_camera.zip'
url = 'http://localhost:8000/api/backup/restore'

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
