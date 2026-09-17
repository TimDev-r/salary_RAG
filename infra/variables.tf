variable "subscription_id" {
  description = "Azure for Students subscription."
  type        = string
}

variable "location" {
  description = <<-EOT
    Azure region.

    NOT westeurope. A new Azure for Students subscription is restricted to a
    subset of regions; West Europe, North Europe, France Central, UK South and
    East US all returned 403 RequestDisallowedByAzure -- "the selected region
    is currently not accepting new customers". Probed live, not assumed.

    Allowed for this subscription: germanywestcentral, swedencentral.
    Frankfurt is the closer of the two to Austria and keeps the data in the EU.
  EOT
  type    = string
  default = "germanywestcentral"
}

variable "prefix" {
  description = "Name prefix for every resource."
  type        = string
  default     = "atkv"
}

variable "image" {
  description = <<-EOT
    Container image. Defaults to a tiny public placeholder so the first apply
    succeeds before anything has been pushed to ghcr.io.

    After that, CI owns this value -- see the lifecycle block in main.tf.
  EOT
  type    = string
  default = "mcr.microsoft.com/k8se/quickstart:latest"
}

# --------------------------------------------------------------------------
# THE COST CONTROLS. Both are variables ONLY so they can be validated; the
# validation is the point, not the configurability.
# --------------------------------------------------------------------------

variable "min_replicas" {
  description = "Idle replica count. MUST be 0 -- see the validation."
  type        = number
  default     = 0

  validation {
    # One 0.5-vCPU replica running continuously consumes about 1,296,000
    # vCPU-seconds a month against a free grant of 180,000 -- roughly seven
    # times over, billed hourly whether or not anyone calls the service.
    #
    # This is the single most expensive character in the repository, so
    # changing it fails `terraform plan` rather than quietly costing money.
    condition     = var.min_replicas == 0
    error_message = "min_replicas must be 0. Any other value bills 24/7 at the idle rate and breaks the EUR 0 constraint this project is built around."
  }
}

variable "max_replicas" {
  description = "Scale ceiling. Caps the blast radius of a traffic spike."
  type        = number
  default     = 1

  validation {
    condition     = var.max_replicas >= 1 && var.max_replicas <= 2
    error_message = "max_replicas must be 1 or 2. A crawler hitting a public endpoint should not be able to scale this into the paid tier."
  }
}

variable "log_retention_days" {
  description = "Log Analytics retention. 30 is the minimum billable-free period."
  type        = number
  default     = 30

  validation {
    condition     = var.log_retention_days <= 30
    error_message = "Log Analytics bills on ingestion and retention beyond 30 days. Keep it at the floor."
  }
}
