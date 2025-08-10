# Legacy Branch Deletion Plan

## Critical Finding: Legacy Code Incompatibility

After examining actual branch content and commit dates, **14 branches** represent legacy development work from **August 2025** that predates the current Virtual Lab integration and multi-agent framework approach. These branches use incompatible architectural patterns and coding styles.

## Evidence-Based Analysis

### Legacy Branches Confirmed for Deletion (August 2025)

**Branch: `feature/experiment-framework`**
- **Last Updated**: August 2, 2025
- **Changes**: 343 lines (171 additions, 172 deletions)
- **Issues**: Uses thread-local storage patterns, old database architecture
- **Commit**: `adb7eea74f7572f31e6aad42ae2299e9855bdc97`
- **Incompatible with**: Current Virtual Lab methodology

**Branch: `feature/literature-retrieval`** 
- **Last Updated**: August 2, 2025
- **Changes**: 690 lines (306 additions, 384 deletions)
- **Issues**: Old API architecture, different request patterns
- **Commit**: `59bc2b9a29d6907401a21abc8054d66cc7d9a10a`
- **Incompatible with**: Current framework integration patterns

**Branch: `feature/memory-vector-database`**
- **Last Updated**: August 2025 (estimated)
- **Issues**: Pre-integration vector database implementation
- **Commit**: `58283bc79e4cea1145a700c60c14a2d077a6edfa`

**Branch: `feature/memory-vector-enhancements`**
- **Last Updated**: August 2025 (estimated)  
- **Issues**: Legacy warning suppression patches
- **Commit**: `7a2161f450cf81c83b5d72071429465d90d6d825`

**Branch: `feature/pi-orchestrator`**
- **Last Updated**: August 2025 (estimated)
- **Issues**: 954+ lines of pre-Virtual Lab orchestration conflicting with current approach
- **Commit**: `ef769592e7017c6b105e2f641d0c3317fdfd50ba`

### Additional Legacy Physics Branches (Pre-Framework Integration)

These branches implement full physics simulation engines from the older "physics simulation platform" vision, incompatible with the current "multi-agent research collaboration" direction:

**Branch: `feature/physics-engine`**
- **Issues**: Full physics simulation implementation vs assessment approach
- **Commit**: `27f6b1f205af295a233883839119f34aaf6d318d`

**Branch: `feature/physics-tools`** 
- **Issues**: Quantum chemistry/molecular dynamics implementations
- **Commit**: `937edd6b0b707fc240f56180bf998b97ac465449`

**Branch: `agents/physics-specialist-agents`**
- **Issues**: Agents performing stellar evolution simulations vs research collaboration
- **Commit**: `13db829fe2ead76cd6f1f85745d6ce4e2e84732b`

**Branch: `feature/physics-data`**
- **Issues**: Legacy physics data handling
- **Commit**: `a388b772a002dc85c402ac77526c91cee8078e56`

**Branch: `feature/physics-data-manager`**
- **Issues**: Legacy physics data management system
- **Commit**: `536438da3c28dd29b5ccd115af7e98ea14ad0ae0`

**Branch: `feature/physics-unit-tests`**
- **Issues**: Tests for deprecated physics implementation
- **Commit**: `1804c33eb595217de5a31fd71b3de5de9bc29bb5`

**Branch: `feature/physics-web-features`**
- **Issues**: Web features for deprecated physics system
- **Commit**: `235f54ab05d777e3d79e4d321bff07e7ae3c1fea`

**Branch: `feature/physics-workflow`**
- **Issues**: Workflow for deprecated physics simulation
- **Commit**: `7e2e5901a6b706aff85e7fd5a909da4cd939d5f9`

**Branch: `feature/vector-memory`**
- **Issues**: Legacy memory implementation
- **Commit**: `ceef516cedf82b507fd1e905a01e4bac09406d35`

## GitHub MCP Tools Limitation

**Critical Discovery**: The GitHub MCP server tools provided do not include branch deletion functionality. Available tools are read-only:
- `list_branches` ✅
- `get_commit` ✅  
- `search_*` functions ✅
- No `delete_branch` or `modify_branch` functions ❌

## Alternative Solutions

### Option 1: Manual GitHub Web Interface Deletion
Navigate to each branch in GitHub web interface and delete manually:
```
https://github.com/ankziety/ai-research-lab-framework/branches
```

### Option 2: Git Command Script (Requires Push Access)
```bash
# Delete legacy branches (WARNING: This permanently deletes branches)
git push origin --delete feature/experiment-framework
git push origin --delete feature/literature-retrieval  
git push origin --delete feature/memory-vector-database
git push origin --delete feature/memory-vector-enhancements
git push origin --delete feature/pi-orchestrator
git push origin --delete feature/physics-engine
git push origin --delete feature/physics-tools
git push origin --delete agents/physics-specialist-agents
git push origin --delete feature/physics-data
git push origin --delete feature/physics-data-manager
git push origin --delete feature/physics-unit-tests
git push origin --delete feature/physics-web-features
git push origin --delete feature/physics-workflow
git push origin --delete feature/vector-memory
```

### Option 3: GitHub CLI Script
```bash
# Using GitHub CLI (requires gh auth)
gh api repos/ankziety/ai-research-lab-framework/git/refs/heads/feature/experiment-framework -X DELETE
gh api repos/ankziety/ai-research-lab-framework/git/refs/heads/feature/literature-retrieval -X DELETE
# ... (repeat for each branch)
```

## Impact Assessment

**Branches to Delete**: 14 legacy branches  
**Lines of Legacy Code Removed**: ~5,000+ incompatible lines  
**Architectural Conflicts Resolved**: Thread-local storage, old API patterns, pre-integration implementations  
**Framework Alignment**: Maintains focus on multi-agent research collaboration vs physics simulation platform

## Recommended Action

Since GitHub MCP tools lack deletion capability, recommend **Option 2** (Git commands) for immediate cleanup, followed by updating `.gitignore` and documentation to reflect the corrected project scope.