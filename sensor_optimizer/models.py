# sensor_optimizer/models.py
from django.db import models

class Feedback(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    name = models.CharField(max_length=120, blank=True)
    email = models.EmailField(blank=True)
    rating = models.PositiveSmallIntegerField(default=5)  # 1–5
    message = models.TextField()

    page = models.CharField(max_length=200, blank=True)   # where user was when sending
    user_agent = models.CharField(max_length=300, blank=True)

    # simple anti-spam honeypot (should stay empty)
    website = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Feedback {self.created_at:%Y-%m-%d %H:%M} ({self.rating}/5)"