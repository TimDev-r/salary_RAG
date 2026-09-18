# ---------------------------------------------------------------------------
# AT-KV assistant -- Azure Container Apps, consumption plan, scale-to-zero.
#
# FOUR RESOURCES, and the reason there are only four:
#
#   resource group             a folder with a lifecycle           EUR 0
#   log analytics workspace    required by the environment          EUR 0 idle,
#                                                                   bills on GB ingested
#   container apps environment the network/DNS boundary             EUR 0
#   container app              the service itself                   EUR 0 at zero replicas
#
# Everything that bills on Azure bills for EXISTING, not for being used. The
# only reason this costs nothing is that at min_replicas = 0 there is nothing
# in existence between requests.
#
# Deliberately absent: a container registry (ghcr.io is free), a search service
# (the FAISS index ships inside the image), a VNet or private endpoint (real
# daily cost on an idle environment), and any provisioned compute.
#
# STATE IS LOCAL AND GITIGNORED. A remote backend needs a storage account, and
# a storage account is pennies a month rather than zero. The trade is made
# explicitly: infrastructure is applied from one laptop, deliberately; CI never
# runs Terraform and only updates the image tag. In a team this would be an
# Azure Storage backend with blob-lease locking, and the few cents would be
# obviously worth it.
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.9"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id

  # Register only what we use. The default registers a broad set, which is slow
  # on a fresh subscription and enables namespaces we have no intention of
  # touching.
  resource_provider_registrations = "none"
  resource_providers_to_register  = ["Microsoft.App", "Microsoft.OperationalInsights"]
}

resource "azurerm_resource_group" "rg" {
  name     = "rg-${var.prefix}"
  location = var.location

  tags = {
    project     = "atkv"
    cost_target = "eur-0"
    managed_by  = "terraform"
  }
}

# Not optional: a Container Apps environment requires a workspace. It is free
# to exist and bills per GB ingested, which is why the service logs terse JSON
# and third-party INFO chatter is silenced in atkv/logging.py.
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "log-${var.prefix}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days
  tags                = azurerm_resource_group.rg.tags
}

resource "azurerm_container_app_environment" "env" {
  name                       = "cae-${var.prefix}"
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
  tags                       = azurerm_resource_group.rg.tags
}

resource "azurerm_container_app" "app" {
  name                         = "ca-${var.prefix}"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.rg.name
  revision_mode                = "Single"
  tags                         = azurerm_resource_group.rg.tags

  template {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    container {
      name   = "atkv"
      image  = var.image
      # 1 vCPU / 2 GiB, not 0.5 / 1 GiB.
      #
      # The app peaks around 0.7 GiB loading the embedder, the translation
      # model and the index, which left too little headroom under a 1 GiB cap
      # and the container was killed with no traceback. Half a vCPU also made
      # startup slow enough to lose a race with the health probes.
      #
      # This does NOT change the idle cost, which is what matters: at
      # min_replicas 0 nothing exists between requests. It doubles consumption
      # per second of actual runtime, against a monthly free grant of 180,000
      # vCPU-seconds -- about 50 hours of real request handling.
      cpu    = 1.0
      memory = "2Gi"

      env {
        name  = "OMP_NUM_THREADS"
        value = "1"
      }
      env {
        name  = "ATKV_RERANK"
        value = "0" # the cross-encoder costs ~1400ms and no longer improves R@5
      }

      # A STARTUP PROBE is the right tool for a slow cold start, rather than a
      # long liveness initial_delay (which Azure caps at 60s anyway). While it
      # is failing, liveness is not evaluated at all -- so a 2.6 GB image pull
      # plus two model loads cannot be mistaken for a hung process and killed.
      # 20 x 15s gives five minutes before the platform gives up.
      startup_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/healthz"
        interval_seconds        = 15
        failure_count_threshold = 20
      }

      readiness_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/healthz"
        # Generous: a cold start pulls a 2.6 GB image and loads two models.
        # Probes that fire too early turn a slow start into a crash loop.
        initial_delay           = 20
        interval_seconds        = 15
        failure_count_threshold = 12
      }

      liveness_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/healthz"
        initial_delay           = 60
        interval_seconds        = 30
        failure_count_threshold = 5
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  lifecycle {
    # CI OWNS THE IMAGE TAG, TERRAFORM OWNS EVERYTHING ELSE.
    #
    # Without this, the next `terraform apply` would roll the app back to
    # var.image -- undoing whatever CI last deployed, silently, as a side
    # effect of an unrelated infrastructure change. That is the classic
    # IaC-versus-CD conflict, and ignore_changes is how the boundary is drawn.
    ignore_changes = [template[0].container[0].image]
  }
}
