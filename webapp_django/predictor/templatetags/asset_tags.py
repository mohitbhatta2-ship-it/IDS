"""
Cache-busting static URLs for development.

Django's dev server lets the browser cache static files, so an edited CSS or JS
file can keep serving the old version until a manual hard reload -- which looks
exactly like "my change did nothing". Appending the file's modification time
makes the URL change whenever the file does.

In production WhiteNoise already hashes filenames, so this adds nothing there
and the plain URL is used.
"""

from pathlib import Path

from django import template
from django.conf import settings
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@register.simple_tag
def static_v(path: str) -> str:
    url = static(path)

    if not settings.DEBUG:
        return url

    found = finders.find(path)
    if not found:
        return url

    try:
        stamp = int(Path(found).stat().st_mtime)
    except OSError:
        return url

    return f"{url}?v={stamp}"
