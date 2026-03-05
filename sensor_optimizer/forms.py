# sensor_optimizer/forms.py
from django import forms
from .models import Feedback


class FeedbackForm(forms.ModelForm):
    # nicer rating widget (1–5)
    rating = forms.ChoiceField(
        choices=[(i, str(i)) for i in range(1, 6)],
        initial="5",
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Rating",
    )

    class Meta:
        model = Feedback
        fields = ["name", "email", "rating", "message", "website"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "email": forms.EmailInput(attrs={"class": "form-control"}),
            "message": forms.Textarea(attrs={"rows": 5, "class": "form-control"}),
            # honeypot (hidden in template)
            "website": forms.TextInput(attrs={"autocomplete": "off", "class": "form-control"}),
        }

    def clean_website(self):
        v = (self.cleaned_data.get("website") or "").strip()
        if v:
            raise forms.ValidationError("Spam detected.")
        return ""