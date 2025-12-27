# Database Population via Helm Chart

This directory contains Helm templates for populating Milvus and MongoDB databases with initial data before deploying the main application.

## Overview

The database population feature allows you to:
1. Create a PVC for storing database dumps
2. Upload data manually via an upload pod
3. Run a Job that populates both Milvus and MongoDB from the uploaded data

All resources are managed through Helm values, eliminating the need for manual YAML editing.

## Configuration

Enable database population in your `values.yaml`:

```yaml
dbPopulation:
  enabled: true  # Enable database population resources
  pvc:
    size: 20Gi
    storageClass: ""  # Uses global storageClass.name if empty
  resources:
    requests:
      memory: 2Gi
    limits:
      memory: 4Gi
```

## Usage Workflow

### Step 1: Install Helm Chart with Database Population Enabled

```bash
helm install <release-name> ./helm/aiod_enhanced_interaction \
  -f values.yaml \
  --set dbPopulation.enabled=true \
  --set app.enabled=false
```

This creates:
- PVC for database dumps
- Upload pod (if enabled)
- Population job (ready to run)

### Step 2: Upload Database Dumps

Prepare your local data directory:
```
local-data/
├── mongo-dumps/
│   ├── assetCollections.json
│   └── assetsForMetadataExtraction.json
└── embeddings/
    └── (your .npz and .json files)
```

Upload using the helper script:
```bash
./scripts/upload-data-helm.sh <release-name> <namespace> ./local-data
```

Or manually:
```bash
POD_NAME="<release-name>-upload-data"
NAMESPACE="<namespace>"

# Create directories
kubectl exec -it $POD_NAME -n $NAMESPACE -- mkdir -p /data/mongo-dumps /data/embeddings

# Upload files
kubectl cp ./local-data/mongo-dumps/assetCollections.json $NAMESPACE/$POD_NAME:/data/mongo-dumps/
kubectl cp ./local-data/mongo-dumps/assetsForMetadataExtraction.json $NAMESPACE/$POD_NAME:/data/mongo-dumps/
kubectl cp ./local-data/embeddings/ $NAMESPACE/$POD_NAME:/data/embeddings/
```

### Step 3: Run Population Job

The job is created but doesn't run automatically. You can trigger it:

```bash
# Option 1: Create a new job from the template
kubectl create job --from=job/<release-name>-populate-databases \
  <release-name>-populate-databases-manual \
  -n <namespace>

# Option 2: Delete and recreate via Helm upgrade
helm upgrade <release-name> ./helm/aiod_enhanced_interaction \
  -f values.yaml \
  --set dbPopulation.enabled=true
```

### Step 4: Monitor Population

```bash
# Watch job status
kubectl get job <release-name>-populate-databases -w

# Check Milvus population logs
kubectl logs -f job/<release-name>-populate-databases -c populate-milvus

# Check MongoDB population logs
kubectl logs -f job/<release-name>-populate-databases -c populate-mongodb
```

### Step 5: Deploy Application

Once population is complete, deploy the full application:

```bash
helm upgrade <release-name> ./helm/aiod_enhanced_interaction \
  -f values.yaml \
  --set dbPopulation.enabled=false  # Disable after population
  --set app.enabled=true
```

## Features

- **Automatic Templating**: All service names, namespaces, and secrets are automatically templated
- **Milvus Lite Support**: Handles both Milvus Standalone and Milvus Lite modes
- **Conditional Logic**: Only creates resources when needed (e.g., Milvus wait initContainer only when Milvus is enabled)
- **Resource Management**: Configurable resource limits for population containers
- **PVC Persistence**: Database dumps PVC persists across Helm upgrades

## Troubleshooting

### Upload Pod Not Found

Ensure `dbPopulation.enabled=true` and `dbPopulation.uploadPod.enabled=true` in your values.

### Population Job Fails

1. Check that MongoDB and Milvus services are ready:
   ```bash
   kubectl get svc | grep -E "mongo|milvus"
   ```

2. Verify secrets exist:
   ```bash
   kubectl get secret <release-name>-mongo-secret
   kubectl get secret <release-name>-milvus
   ```

3. Check job logs:
   ```bash
   kubectl describe job <release-name>-populate-databases
   kubectl logs job/<release-name>-populate-databases -c populate-milvus
   kubectl logs job/<release-name>-populate-databases -c populate-mongodb
   ```

### Data Not Found

Verify files were uploaded correctly:
```bash
kubectl exec <release-name>-upload-data -- ls -lah /data/mongo-dumps/
kubectl exec <release-name>-upload-data -- ls -lah /data/embeddings/
```

## Cleanup

After successful population:

```bash
# Delete upload pod (optional)
kubectl delete pod <release-name>-upload-data

# Delete population job (optional, keeps history)
kubectl delete job <release-name>-populate-databases

# Note: PVC is kept by default (helm.sh/resource-policy: keep)
# Delete manually if needed:
# kubectl delete pvc <release-name>-db-dumps-pvc
```
