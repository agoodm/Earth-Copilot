param location string = resourceGroup().location
param tags object = {}

param name string
param sku object = {
  name: 'S0'
}

@description('Deploy AI models')
param deployModels bool = true

// Azure AI Foundry resource (formerly Azure AI Services)
// This provides access to the Model Catalog with multiple models
resource aiFoundry 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  tags: tags
  kind: 'AIServices'  // AIServices provides access to Model Catalog
  sku: sku
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// Deploy GPT-4o model
resource gpt4oDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployModels) {
  parent: aiFoundry
  name: 'gpt-4o'
  sku: {
    name: 'Standard'
    capacity: 10
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-11-20'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
}

// Deploy GPT-4o-mini model (faster, cheaper option)
resource gpt4oMiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployModels) {
  parent: aiFoundry
  name: 'gpt-4o-mini'
  sku: {
    name: 'Standard'
    capacity: 10
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o-mini'
      version: '2024-07-18'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
  dependsOn: [
    gpt4oDeployment
  ]
}

// Note: GPT-5 will be added here when available in Azure AI Foundry Model Catalog
// Additional models from Model Catalog (Llama, Phi, etc.) can be added as needed

output name string = aiFoundry.name
output endpoint string = aiFoundry.properties.endpoint
output id string = aiFoundry.id
