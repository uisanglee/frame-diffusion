"""Bounded recovery of infrastructure errors; never alter HTML or scores."""
import math
import time


def render_with_retry(browser, operation, timing, attempts=3):
    """Retry Chromium capture failures with a fresh context and identical input."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            message = str(error)
            transient = ('Page.captureScreenshot' in message or
                         'Unable to capture screenshot' in message)
            if not transient or attempt == attempts:
                raise
            timing.event('render_retry', 'retry', attempt=attempt,
                         max_attempts=attempts, error=message)
            browser.reset_context()
            time.sleep(.2 * attempt)


def reusable_metric(row, initial_failed, official, webui=False):
    if initial_failed:
        return row.get('evaluation_skipped') == 'initial_generation_failed'
    if any(row.get(key) for key in ('evaluation_error', 'reference_error', 'geometry_error')):
        return False
    metrics = ['webui_box_iou' if webui else 'dom_box_iou']
    if official:
        metrics += ['official_position', 'official_clip']
    return all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
               for key in metrics)
