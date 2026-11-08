# Routing API Reference

The Routing API is a REST service whose endpoints live under the base path /v2. Requests are authenticated with an API key sent in the Authorization header.

Each API key is limited to 600 requests per minute. Requests above the limit receive HTTP 429 together with a Retry-After header. API keys must be rotated every 90 days; expired keys return HTTP 401.

Webhook deliveries are retried five times with exponential backoff before being marked as failed. To make POST requests safe to repeat, send an Idempotency-Key header; the key is remembered for 24 hours.
