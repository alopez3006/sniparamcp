# Self-Hosted Enterprise - Snipara

## Overview

**Snipara Self-Hosted Enterprise** is a custom licensing option for organizations that require on-premises deployment of the Snipara MCP server with full control over their infrastructure.

## Why Self-Host?

Organizations choose self-hosted Snipara for:

- **Data Sovereignty:** Keep all documentation and embeddings within your infrastructure
- **Compliance Requirements:** Meet regulatory requirements (HIPAA, SOC2, FedRAMP)
- **Air-Gapped Environments:** Deploy in networks without internet access
- **Custom Infrastructure:** Integrate with existing PostgreSQL, Redis, monitoring
- **SLA Requirements:** Full control over uptime and performance

## What's Included

### Software & Support

| Feature | Description |
|---------|-------------|
| **Full Source Code** | Complete FastAPI MCP server with all features |
| **Security Features** | Audit logging, anti-scan protection, multi-project ACL |
| **Advanced Embeddings** | bge-large-en-v1.5 (1024 dimensions) |
| **Agent Infrastructure** | Memory system, swarms, multi-agent coordination |
| **Team Features** | Shared context collections, multi-project queries |
| **Quarterly Updates** | Security patches, feature updates, performance improvements |
| **Deployment Support** | Docker images, Kubernetes manifests, Railway/Render configs |
| **Documentation** | Architecture docs, API reference, runbooks |

### Professional Services

| Service | Included |
|---------|----------|
| **Initial Setup** | ✅ 8 hours - Infrastructure assessment, deployment planning |
| **White-Glove Deployment** | ✅ Full deployment to your infrastructure |
| **Custom Integration** | ✅ SSO, existing databases, monitoring tools |
| **Training** | ✅ 4 hours - Admin training, best practices workshop |
| **Ongoing Support** | ✅ Email + Slack support (24-hour SLA) |
| **Quarterly Reviews** | ✅ Performance review, optimization recommendations |

## Pricing

### Base License

**$2,000/month minimum** (billed annually: $24,000/year)

Includes:
- Up to 50,000 queries/month
- Unlimited projects and users
- All features (security, agents, team context)
- Standard support (24-hour response SLA)

### Volume Pricing

| Monthly Queries | Price/Month |
|----------------|-------------|
| 50K | $2,000 |
| 100K | $3,500 |
| 250K | $7,500 |
| 500K | $12,000 |
| 1M+ | Custom |

### Premium Support Add-Ons

| Add-On | Price |
|--------|-------|
| **Priority Support** (4-hour SLA) | +$500/month |
| **24/7 On-Call** (1-hour SLA) | +$2,000/month |
| **Dedicated CSM** | +$3,000/month |
| **Custom Feature Development** | Starting at $10,000/project |

## Infrastructure Requirements

### Minimum Specifications

```yaml
PostgreSQL:
  Version: "14+"
  Storage: "100GB SSD"
  Memory: "4GB RAM"
  Note: "Neon, Supabase, or self-hosted"

Redis:
  Version: "7+"
  Memory: "2GB RAM"
  Note: "For rate limiting and anti-scan protection"

Application Server:
  CPU: "4 cores"
  Memory: "8GB RAM"
  Storage: "20GB"
  Note: "Docker-compatible environment"

Python:
  Version: "3.11+"
  Note: "For FastAPI server"

GPU (Optional):
  Type: "NVIDIA T4 or better"
  Memory: "16GB VRAM"
  Note: "For embedding generation (CPU fallback available)"
```

### Recommended Production Setup

```yaml
Load Balancer:
  Type: "AWS ALB, Cloudflare, or nginx"
  SSL: "Required (Let's Encrypt or custom cert)"

Application Tier:
  Instances: "2+ (auto-scaling)"
  CPU: "8 cores per instance"
  Memory: "16GB RAM per instance"

Database:
  Type: "PostgreSQL with read replicas"
  Storage: "500GB SSD"
  Backup: "Daily automated backups"

Cache:
  Type: "Redis Cluster"
  Nodes: "3+ (for high availability)"

Monitoring:
  Metrics: "Prometheus + Grafana"
  Logs: "ELK Stack or CloudWatch"
  Alerts: "PagerDuty or Opsgenie"

Estimated Cost: $1,000-$3,000/month (infrastructure only)
```

## Deployment Options

### 1. Docker + Docker Compose (Simplest)

- ✅ Single-server deployment
- ✅ Ideal for dev/staging
- ✅ Quick setup (< 1 hour)
- ❌ Limited scalability

### 2. Kubernetes (Most Flexible)

- ✅ Auto-scaling
- ✅ High availability
- ✅ Multi-region support
- ⚠️ Complex setup (requires K8s expertise)

### 3. PaaS (Railway, Render, Fly.io)

- ✅ Managed infrastructure
- ✅ Easy scaling
- ✅ Built-in monitoring
- ⚠️ Higher cost per query

### 4. Custom Cloud (AWS, GCP, Azure)

- ✅ Full control
- ✅ Integration with existing infrastructure
- ✅ Cost optimization
- ⚠️ Requires DevOps team

## Security & Compliance

### Included Security Features

- **Audit Logging:** All API calls logged to PostgreSQL
- **Anti-Scan Protection:** Rate limiting + project enumeration prevention
- **Multi-Project ACL:** Fine-grained access control per project
- **API Key Management:** Team API keys with scoped permissions
- **OAuth Support:** GitHub, Google, custom SAML/OIDC
- **Secrets Management:** Vault, AWS Secrets Manager, Azure Key Vault

### Compliance Support

We provide documentation and architecture guidance for:

- **SOC 2 Type II:** Audit trail, access controls, encryption
- **HIPAA:** BAA available, encryption at rest/in transit
- **GDPR:** Data retention policies, right to deletion
- **ISO 27001:** Security controls, incident response
- **FedRAMP:** Government cloud deployment support

## Comparison: Hosted vs Self-Hosted

| Feature | Hosted Cloud | Self-Hosted Enterprise |
|---------|--------------|------------------------|
| **Setup Time** | 5 minutes | 2-4 weeks |
| **Infrastructure Management** | ✅ Managed by Snipara | ❌ Your responsibility |
| **Data Location** | US/EU regions | ✅ Your infrastructure |
| **Scaling** | ✅ Automatic | Manual or auto-configured |
| **Updates** | ✅ Automatic | Quarterly releases |
| **Cost (50K queries/mo)** | $49/month (Team) | $2,000/month |
| **Support** | Email | Email + Slack + Quarterly reviews |
| **Compliance** | SOC 2 | ✅ Your compliance framework |

## Getting Started

### Step 1: Contact Sales

Email: **sales@snipara.com**

Include:
- Company name and size
- Expected query volume
- Infrastructure preferences (AWS, GCP, Azure, on-prem)
- Compliance requirements
- Timeline for deployment

### Step 2: Scoping Call (30-60 minutes)

We'll discuss:
- ✅ Technical requirements
- ✅ Integration points (SSO, databases, monitoring)
- ✅ Deployment timeline
- ✅ Support needs
- ✅ Custom pricing (if needed)

### Step 3: Contract & Onboarding

- Sign license agreement (annual or multi-year)
- Receive access to private repo + Docker registry
- Schedule deployment kickoff call
- Begin infrastructure assessment

### Step 4: Deployment (2-4 weeks)

**Week 1:** Infrastructure setup + database migration
**Week 2:** Application deployment + SSL configuration
**Week 3:** Integration testing + performance tuning
**Week 4:** Training + production cutover

### Step 5: Production + Ongoing Support

- 24-hour email/Slack support
- Quarterly reviews
- Security patches within 48 hours
- Feature updates every quarter

## FAQ

### Q: Can I try before buying?

**A:** Yes. We offer a 2-week proof-of-concept deployment in your infrastructure. Contact sales for details.

### Q: What if I exceed my query limit?

**A:** Soft limits with 10% overage included. Contact us to upgrade your tier.

### Q: Can I mix hosted and self-hosted?

**A:** Yes. Many customers use hosted for dev/staging and self-hosted for production.

### Q: What happens if I cancel?

**A:** You retain your license until the end of your contract term. No refunds on annual plans.

### Q: Do you support air-gapped environments?

**A:** Yes. We provide offline deployment packages and documentation. Premium support required.

### Q: Can you handle our existing PostgreSQL database?

**A:** Yes. We can integrate with your existing PostgreSQL instance. Requires PostgreSQL 14+.

### Q: What's the difference between this and open source?

**A:** Self-Hosted Enterprise includes:
- Latest security features (audit logging, anti-scan)
- Agent infrastructure (memory, swarms)
- Team features (shared context collections)
- Professional support + deployment assistance
- Quarterly updates and security patches

The open source version (previously available on PyPI) is now deprecated and has critical bugs.

## Contact

**Sales:** sales@snipara.com
**Website:** https://snipara.com/self-hosted
**Documentation:** https://docs.snipara.com/self-hosted

---

*Pricing effective January 2026. Subject to change. All prices in USD.*
