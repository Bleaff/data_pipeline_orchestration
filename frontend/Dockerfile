# Dev-mode container for the neudc dashboard (see ../docker-compose.yml).
#
# Runs `next dev`, not a production build — this is for local iteration against
# either the dockerized `api` service or a host-run pipeline (e.g. the voice
# assistant's embedded control-plane API), not for deploying the dashboard anywhere.
# docker-compose.yml bind-mounts ./frontend over /app so source edits on the host
# take effect without a rebuild; node_modules/.next stay in the image (or a docker
# volume) rather than the bind mount so the container's own (linux) build isn't
# clobbered by the host's.

FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
