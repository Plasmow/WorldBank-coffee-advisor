"""Africa's Talking SMS gateway.

Demo numbers and DEMO_MODE never reach the network: a demo visitor must not
cause a real SMS, and a gateway outage must not break the webhook.
"""

import logging
import os

import httpx

log = logging.getLogger(__name__)
AT_URL = "https://api.sandbox.africastalking.com/version1/messaging"


def send_sms(to, text):
    from app.clock import is_demo  # local: breaks the db -> at_client -> clock cycle

    log.info("sms to %s: %s", to, text)
    if os.environ.get("DEMO_MODE") == "1" or is_demo(to):
        return  # demo traffic never reaches the gateway
    try:
        httpx.post(
            AT_URL,
            data={
                "username": os.environ.get("AT_USERNAME", "sandbox"),
                "to": to,
                "message": text,
                "from": os.environ.get("AT_SHORTCODE", ""),
            },
            headers={
                "apiKey": os.environ.get("AT_API_KEY", ""),
                "Accept": "application/json",
            },
            timeout=10,
        )
    except Exception:  # a dead gateway must never 500 the webhook
        log.exception("africa's talking send failed")
