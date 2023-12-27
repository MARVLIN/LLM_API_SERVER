import requests

url = "http://localhost:8000/file/multiple_upload/"
file_path = "/Users/artemiikhristich/PycharmProjects/drf-file-upload/first/email1.txt"
filename = "email1.txt"
files = {'files': (filename, open(file_path, 'rb'))}

response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)  # Print the entire response text for more details
