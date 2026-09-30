from django.contrib.auth.models import Group


def user_role(request):

    role = "Analyst"

    if request.user.is_authenticated:

        if request.user.is_superuser:
            role = "Admin"

        elif request.user.groups.filter(name="Admin").exists():
            role = "Admin"

    return {
        "user_role": role,
        "is_admin_user": role == "Admin",
    }