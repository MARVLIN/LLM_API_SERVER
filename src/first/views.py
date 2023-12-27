import json
from concurrent.futures import ThreadPoolExecutor

from llama_cpp import Llama
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView

from .models import FileModel, ImageModel
from .serializers import FileSerializer, ImageSerializer, MultipleImageSerializer, MultipleFileSerializer
from .utils import classify_email_and_extract_info, extract_entities


class FileViewSet(viewsets.ModelViewSet):
    queryset = FileModel.objects.all()
    serializer_class = FileSerializer

    @action(detail=False, methods=["POST"])
    def multiple_upload(self, request, *args, **kwargs):
        """Upload multiple files and create objects."""
        serializer = MultipleFileSerializer(data=request.data or None)
        serializer.is_valid(raise_exception=True)
        files = serializer.validated_data.get("file")  # Use "file" instead of "files"

        files_list = []
        for file in files:
            files_list.append(
                FileModel(file=file)
            )
        if files_list:
            FileModel.objects.bulk_create(files_list)

        llm = Llama(model_path="/home/art1x/dev/Mistral/dolphin-2.2.1-mistral-7b.Q5_K_S.gguf",
                    n_ctx=4096, n_gpu_layers=8, chat_format="functionary")

        responses = []

        for file in files:
            # Read the file content
            file_content = file.read().decode("utf-8")

            # Save the file model first
            file_model = FileModel(file=file)
            file_model.save()

            # Call the function to classify the email and extract the information
            result = classify_email_and_extract_info(llm, file_content)

            # Save the entire result as llm_output in the FileModel
            file_model.llm_output = json.dumps(result)  # Convert result to JSON
            file_model.save()  # Save the file model again

            responses.append({"file_id": file_model.id, "llm_output": file_model.llm_output})

        return Response(data={"message": "Files processed successfully", "responses": responses})


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageSerializer

    @action(detail=False, methods=["POST"])
    def multiple_upload(self, request, *args, **kwargs):
        """Upload multiple images and create objects."""
        serializer = MultipleImageSerializer(data=request.data or None)
        serializer.is_valid(raise_exception=True)
        images = serializer.validated_data.get("images")

        images_list = [ImageModel(file=image) for image in images]

        if images_list:
            ImageModel.objects.bulk_create(images_list)

        return Response({"message": "Images processed successfully"})



# WITHOUT DRF UPLOADS

def single_upload(request):
    file = request.FILES.get("file")
    FileModel.objects.create(file=file)

    llm = Llama(model_path="/home/art1x/dev/Mistral/dolphin-2.2.1-mistral-7b.Q5_K_S.gguf", n_ctx=4096,
                n_gpu_layers=8, chat_format="functionary")

    # Extract the email content from the file
    result = classify_email_and_extract_info(llm, file.read().decode("utf-8"))
    print("result", result)
    # save result to FileModel llm_output field
    file.llm_output = result
    file.save()
    print("file.llm_output", file.llm_output)

    return JsonResponse({"message": "Success"})


def multiple_upload(request):
    files = request.FILES.getlist("files")

    files_list = [FileModel(file=files) for file in files]

    if files_list:
        FileModel.objects.bulk_create(files_list)

    llm = Llama(model_path="/home/art1x/dev/Mistral/dolphin-2.2.1-mistral-7b.Q5_K_S.gguf",
                n_ctx=4096, n_gpu_layers=8, chat_format="functionary")

    responses = []
    for file_model in files_list:
        try:
            file_content = file_model.file.read().decode("utf-8")
            result = classify_email_and_extract_info(llm, file_content)
            file_model.llm_output = result
            file_model.save()
            responses.append(file_model.llm_output)
        except Exception as e:
            responses.append(f"Error processing file: {str(e)}")

    return JsonResponse({"message": "Files processed successfully", "responses": responses})



def index(request):
    return render(template_name="index.html", request=request)


class MultipleUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = MultipleFileSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        files = serializer.validated_data.get("files")

        files_list = [FileModel(file=file) for file in files]

        if files_list:
            FileModel.objects.bulk_create(files_list)

        llama_model = Llama(model_path="/home/art1x/dev/Mistral/dolphin-2.2.1-mistral-7b.Q5_K_S.gguf", n_ctx=4096, n_gpu_layers=8, chat_format="functionary")

        # Process files in parallel
        process_files_parallel(llama_model, files_list)

        return Response({"message": "Files processed successfully"})


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

