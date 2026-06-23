import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
from .models import Document
from .rag_engine import process_and_store_document, query_documents

class UploadPDFView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Ensure the media/pdfs directory exists
        save_path = os.path.join(settings.MEDIA_ROOT, 'pdfs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save file locally
        file_name = default_storage.save(f"pdfs/{file_obj.name}", file_obj)
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        
        # Save record to database
        doc = Document.objects.create(
            user=request.user,
            file=file_name,
            filename=file_obj.name
        )
        
        # Process and Embed
        try:
            process_and_store_document(file_path, request.user.id)
            return Response({'message': 'Document uploaded and processed successfully', 'id': doc.id}, status=status.HTTP_201_CREATED)
        except Exception as e:
            # Log the actual python error so you can see it in the terminal
            print(f"ERROR DURING UPLOAD: {e}") 
            return Response({'error': f'Processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class QueryPDFView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        question = request.data.get('question')
        if not question:
            return Response({'error': 'Question required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            answer = query_documents(question, request.user.id)
            return Response({'answer': answer}, status=status.HTTP_200_OK)
        except Exception as e:
            # Log the actual python error so you can see it in the terminal
            print(f"ERROR DURING QUERY: {e}") 
            return Response({'error': f'Query failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)