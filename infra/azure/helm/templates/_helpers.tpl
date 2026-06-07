{{/*
Common labels applied to every resource in this chart.
*/}}
{{- define "reprolab-aks.labels" -}}
app.kubernetes.io/name: reprolab-aks
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Selector labels — stable subset used in matchLabels.
*/}}
{{- define "reprolab-aks.selectorLabels" -}}
app.kubernetes.io/name: reprolab-aks
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Pod labels required for Azure Workload Identity token injection.
Must be present on every pod that calls DefaultAzureCredential (Job pods).
*/}}
{{- define "reprolab-aks.workloadIdentityPodLabel" -}}
azure.workload.identity/use: "true"
{{- end }}
