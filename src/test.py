import requests

url = "http://localhost:8000/file/multiple_upload/"
file_path = "/Users/artemiikhristich/PycharmProjects/JobiFy-2/categorization/llm/email1.txt"
filename = "email1.txt"
files = {'file': (filename, open(file_path, 'rb'))}

response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.status_code)  # Print the entire response text for more details
    print(response.text)
