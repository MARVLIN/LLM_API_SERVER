import json

from llama_cpp import Llama
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import render
from django.http import JsonResponse

from .models import FileModel, ImageModel
from .serializers import FileSerializer, ImageSerializer, MultipleImageSerializer, MultipleFileSerializer
from .utils import classify_email_and_extract_info


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




