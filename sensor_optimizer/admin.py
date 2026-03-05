from django.contrib import admin
from .models import Feedback

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("created_at", "rating", "name", "email", "page")
    search_fields = ("name", "email", "message", "page")
    list_filter = ("rating", "created_at")