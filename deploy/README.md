# Deploy

Deployment configuration for Chart2CSV API.

## Files

| File | Description |
|------|-------------|
| `deploy.sh` | Deployment script for Hetzner server |
| `nginx.conf` | Nginx reverse proxy configuration |

## Deployment

```bash
./deploy/deploy.sh
```

This will:
1. Sync files to server
2. Configure Nginx
3. Build and start Docker container
4. Setup SSL with Let's Encrypt
