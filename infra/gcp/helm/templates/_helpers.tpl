{{/*
Common labels applied to every resource in this chart.
*/}}
{{- define "reprolab-gke.labels" -}}
app.kubernetes.io/name: reprolab-gke
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Selector labels — stable subset used in matchLabels.
*/}}
{{- define "reprolab-gke.selectorLabels" -}}
app.kubernetes.io/name: reprolab-gke
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
