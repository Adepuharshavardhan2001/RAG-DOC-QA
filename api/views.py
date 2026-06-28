import os
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.core.files.storage import default_storage
from django.conf import settings
from .models import Document
from .rag_engine import process_and_store_document, query_documents

logger = logging.getLogger(__name__)


class UploadPDFView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file_obj = request.FILES.get('file')

        if not file_obj:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        if not file_obj.name.lower().endswith('.pdf'):
            return Response({'error': 'Only PDF files are allowed'}, status=status.HTTP_400_BAD_REQUEST)

        if file_obj.size > 10 * 1024 * 1024:
            return Response({'error': 'File size must be under 10MB'}, status=status.HTTP_400_BAD_REQUEST)

        save_path = os.path.join(settings.MEDIA_ROOT, 'pdfs')
        os.makedirs(save_path, exist_ok=True)

        file_name = default_storage.save(f"pdfs/{file_obj.name}", file_obj)
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        doc = Document.objects.create(
            user=request.user,
            file=file_name,
            filename=file_obj.name
        )

        try:
            result = process_and_store_document(file_path, request.user.id)
            logger.info(f"Document processed: user={request.user.id}, chunks={result['chunks']}")
            return Response({
                'message': 'Document uploaded and processed successfully',
                'id': doc.id,
                'chunks': result['chunks']
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            doc.delete()
            logger.error(f"Processing failed: {e}", exc_info=True)
            return Response({'error': f'Processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class QueryPDFView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        question = request.data.get('question', '').strip()

        if not question:
            return Response({'error': 'Question is required'}, status=status.HTTP_400_BAD_REQUEST)

        if len(question) > 500:
            return Response({'error': 'Question must be under 500 characters'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            answer = query_documents(question, request.user.id)
            logger.info(f"Query answered for user {request.user.id}")
            return Response({'answer': answer}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return Response({'error': 'Query processing failed.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)