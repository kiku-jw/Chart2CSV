#!/bin/bash
# Deploy Chart2CSV API to Hetzner server
# Usage: ./deploy.sh

set -e

SERVER="root@37.27.38.186"
DEPLOY_DIR="/opt/chart2csv"

echo "🚀 Deploying Chart2CSV API to $SERVER..."

# Create deploy directory on server
echo "📁 Creating directory..."
ssh $SERVER "mkdir -p $DEPLOY_DIR"

# Sync files
echo "📤 Syncing files..."
rsync -avz --exclude 'venv*' --exclude '.git' --exclude '__pycache__' \
    --exclude '*.pyc' --exclude 'fixtures' --exclude 'docs' --exclude 'wiki' \
    --exclude 'examples' --exclude '*.png' --exclude '*.jpg' \
    ./ $SERVER:$DEPLOY_DIR/

# SSH into server and run deployment
echo "🐳 Building and starting Docker..."
ssh $SERVER << 'ENDSSH'
cd /opt/chart2csv

# Copy nginx config
cp deploy/nginx.conf /etc/nginx/sites-available/chart2csv.kikuai.dev

# Enable nginx site
ln -sf /etc/nginx/sites-available/chart2csv.kikuai.dev /etc/nginx/sites-enabled/

# Get SSL certificate if not exists
if [ ! -f /etc/letsencrypt/live/chart2csv.kikuai.dev/fullchain.pem ]; then
    echo "🔐 Getting SSL certificate..."
    # First start nginx without SSL to get certificate
    cat > /etc/nginx/sites-available/chart2csv.kikuai.dev << 'NGINX'
server {
    listen 80;
    server_name chart2csv.kikuai.dev;
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 200 'Chart2CSV API - Getting SSL...';
    }
}
NGINX
    nginx -t && systemctl reload nginx
    certbot certonly --webroot -w /var/www/certbot -d chart2csv.kikuai.dev --non-interactive --agree-tos -m admin@kikuai.dev
    # Restore full config
    cp deploy/nginx.conf /etc/nginx/sites-available/chart2csv.kikuai.dev
fi

# Build and start Docker container
echo "🐳 Starting Docker containers..."
docker compose down 2>/dev/null || true
docker compose build --no-cache
docker compose up -d

# Reload nginx
nginx -t && systemctl reload nginx

# Wait for container to be ready
sleep 5

# Check health
echo "🏥 Checking health..."
curl -s http://localhost:8010/health || echo "Container starting..."

echo ""
echo "✅ Deployment complete!"
echo "🌐 API available at: https://chart2csv.kikuai.dev"
echo "📖 Docs available at: https://chart2csv.kikuai.dev/docs"
ENDSSH

echo "🎉 Done!"
