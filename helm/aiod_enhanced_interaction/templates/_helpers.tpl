{{/*
Expand the name of the chart.
*/}}
{{- define "aiod-enhanced-interaction.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "aiod-enhanced-interaction.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "aiod-enhanced-interaction.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aiod-enhanced-interaction.labels" -}}
helm.sh/chart: {{ include "aiod-enhanced-interaction.chart" . }}
{{ include "aiod-enhanced-interaction.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "aiod-enhanced-interaction.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aiod-enhanced-interaction.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
App component labels
*/}}
{{- define "aiod-enhanced-interaction.app.labels" -}}
app: app
{{ include "aiod-enhanced-interaction.labels" . }}
{{- end }}

{{/*
App component selector labels
*/}}
{{- define "aiod-enhanced-interaction.app.selectorLabels" -}}
app: app
{{ include "aiod-enhanced-interaction.selectorLabels" . }}
{{- end }}

{{/*
Mongo component labels
*/}}
{{- define "aiod-enhanced-interaction.mongo.labels" -}}
app: mongo
{{ include "aiod-enhanced-interaction.labels" . }}
{{- end }}

{{/*
Mongo component selector labels
*/}}
{{- define "aiod-enhanced-interaction.mongo.selectorLabels" -}}
app: mongo
{{ include "aiod-enhanced-interaction.selectorLabels" . }}
{{- end }}

{{/*
Ollama component labels
*/}}
{{- define "aiod-enhanced-interaction.ollama.labels" -}}
app: ollama
{{ include "aiod-enhanced-interaction.labels" . }}
{{- end }}

{{/*
Ollama component selector labels
*/}}
{{- define "aiod-enhanced-interaction.ollama.selectorLabels" -}}
app: ollama
{{ include "aiod-enhanced-interaction.selectorLabels" . }}
{{- end }}

{{/*
App fullname
*/}}
{{- define "aiod-enhanced-interaction.app.fullname" -}}
{{- printf "%s-app" (include "aiod-enhanced-interaction.fullname" .) }}
{{- end }}

{{/*
Mongo fullname
*/}}
{{- define "aiod-enhanced-interaction.mongo.fullname" -}}
{{- printf "%s-mongo" (include "aiod-enhanced-interaction.fullname" .) }}
{{- end }}

{{/*
Ollama fullname
*/}}
{{- define "aiod-enhanced-interaction.ollama.fullname" -}}
{{- printf "%s-ollama" (include "aiod-enhanced-interaction.fullname" .) }}
{{- end }}

{{/*
Milvus fullname (service name from dependency chart)
*/}}
{{- define "aiod-enhanced-interaction.milvus.fullname" -}}
{{- printf "%s-milvus" .Release.Name }}
{{- end }}
