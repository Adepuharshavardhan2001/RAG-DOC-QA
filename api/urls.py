from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import UploadPDFView, QueryPDFView

urlpatterns = [
    path('auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('upload/', UploadPDFView.as_view(), name='upload_pdf'),
    path('query/', QueryPDFView.as_view(), name='query_pdf'),
]