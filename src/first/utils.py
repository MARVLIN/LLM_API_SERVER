import json
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
