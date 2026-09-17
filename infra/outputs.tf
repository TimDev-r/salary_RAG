output "app_url" {
  description = "Public HTTPS endpoint. /docs serves the OpenAPI UI."
  value       = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}

output "resource_group" {
  value = azurerm_resource_group.rg.name
}

output "container_app_name" {
  description = "Used by the deploy workflow: az containerapp update -n <this>."
  value       = azurerm_container_app.app.name
}

output "cost_guards" {
  description = "Echoed so a plan makes the cost-critical settings visible."
  value = {
    min_replicas      = azurerm_container_app.app.template[0].min_replicas
    max_replicas      = azurerm_container_app.app.template[0].max_replicas
    log_retention_day = azurerm_log_analytics_workspace.logs.retention_in_days
  }
}
