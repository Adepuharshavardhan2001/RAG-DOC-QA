from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError


def validate_pdf(value):
    if not value.name.lower().endswith('.pdf'):
        raise ValidationError('Only PDF files are allowed')


class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    file = models.FileField(upload_to='pdfs/', validators=[validate_pdf])
    filename = models.CharField(max_length=255)
    size = models.IntegerField(default=0)
    chunks = models.IntegerField(default=0)
    processed = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.user.username} - {self.filename}"