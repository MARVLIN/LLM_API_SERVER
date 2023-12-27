from concurrent.futures import ThreadPoolExecutor

from llama_cpp import Llama
from .models import FileModel


def extract_entities(response_data, email_content):
    entities = ["category", "job_title", "company_name", "salary", "location", "links", "remote"]
    print("response_data", type(response_data))
    return response_data


def classify_email_and_extract_info(llama_model, email_content):
    system_message = "Classify the email into APPLIED(Applicant received ), IN_PROGRESS, SUCCESS, REJECTED, OTHER & Extract job application information"

    try:
        result = llama_model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": email_content}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "ClassifyEmailAndExtractInfo",
                        "parameters": {
                            "type": "object",
                            "title": "ClassifyEmailAndExtractInfo",
                            "properties": {
                                "category": {"title": "Category", "type": "string"},
                                "job_title": {"title": "Job Title", "type": "string"},
                                "company_name": {"title": "Company Name", "type": "string"},
                                "salary": {"title": "Salary", "type": "integer"},
                                "location": {"title": "Location", "type": "string"},
                                "links": {"title": "Links", "type": "list"},
                                "remote": {"title": "Remote", "type": "bool"}
                            },
                        }
                    }
                }
            ]
        )

        # Check the type of result and handle accordingly
        if isinstance(result, dict):
            # Extract entities from the result
            extracted_entities = extract_entities(result, email_content)

            # Create a new FileModel instance to save to the database
            file_model = FileModel(llm_output=extracted_entities)
            file_model.save()

            return "Success"
        else:
            # If result is not a dictionary, set llm_output to the entire result string
            file_model = FileModel(llm_output=str(result))
            file_model.save()

            return "Success"

    except Exception as e:
        return f"Error: {str(e)}"


def classify_and_save(file_model, llama_model):
    # Read the file content
    file_content = file_model.file.read().decode("utf-8")

    # Reset the file pointer to the beginning of the file
    file_model.file.seek(0)

    result = classify_email_and_extract_info(llama_model, file_content)

    extracted_entities = extract_entities(result, file_content)

    if isinstance(result, dict):
        # Extract entities from the result
        llm_output = extracted_entities.get("llm_output", None)
        llm_output = str(llm_output) if llm_output is not None else None
    else:
        # If result is not a dictionary, set llm_output to the entire result string
        llm_output = str(result)

    return llm_output


def process_files_parallel(llama_model, files_list):
    with ThreadPoolExecutor(max_workers=None) as executor:
        # Use executor.map to apply the function in parallel
        file_contents = [file_model.file.read().decode("utf-8") for file_model in files_list]
        results = executor.map(lambda file_content: classify_and_save(file_content, llama_model), file_contents)

        for file_model, llm_output in zip(files_list, results):
            file_model.llm_output = llm_output
            file_model.save()




# Example usage
if __name__ == "__main__":
    email = """ 

    """
    # Example usage
    llm = Llama(model_path="/home/art1x/dev/Mistral/dolphin-2.2.1-mistral-7b.Q5_K_S.gguf", n_ctx=4096, n_gpu_layers=8,
                chat_format="functionary")

    # Usage
    result = classify_email_and_extract_info(llm, email)
    print(result)
