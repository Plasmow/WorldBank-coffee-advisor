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

    payload = {
        "username": os.environ.get("AT_USERNAME", "sandbox"),
        "to": to,
        "message": text,
    }
    # Only when we actually have one. The sandbox answers InvalidSenderId to an
    # empty sender id and accepts the message only when the field is absent.
    shortcode = os.environ.get("AT_SHORTCODE", "").strip()
    if shortcode:
        payload["from"] = shortcode

    try:
        response = httpx.post(
            AT_URL,
            data=payload,
            headers={"apiKey": os.environ.get("AT_API_KEY", ""), "Accept": "application/json"},
            timeout=10,
        )
    except Exception:  # a dead gateway must never 500 the webhook
        log.exception("africa's talking send failed")
        return

    _check(response)


def _check(response):
    """Africa's Talking refuses a message with 201 Created and the reason in
    the body, so the status code alone tells you nothing."""
    if response.status_code >= 300:
        log.error(
            "africa's talking refused the send: %s %s", response.status_code, response.text[:300]
        )
        return
    try:
        data = response.json()["SMSMessageData"]
    except Exception:
        log.error("africa's talking sent back something unreadable: %s", response.text[:300])
        return
    if not data.get("Recipients"):
        log.error("africa's talking delivered to nobody: %s", data.get("Message", response.text[:200]))
