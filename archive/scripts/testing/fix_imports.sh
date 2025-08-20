#!/bin/bash
# Fix all imports from moved modules

echo "Fixing imports from moved modules..."

# Fix config imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.config import/from brain_go_brrr.config import/g' {} \;
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from \.\.core\.config import/from ..config import/g' {} \;
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.abnormality_config import/from brain_go_brrr.config import/g' {} \;

# Fix logger imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.logger import/from brain_go_brrr.infra.logger import/g' {} \;

# Fix exceptions imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.exceptions import/from brain_go_brrr.domain.exceptions import/g' {} \;

# Fix channels imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.channels import/from brain_go_brrr.domain.channels import/g' {} \;

# Fix quality imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.quality import/from brain_go_brrr.domain.quality import/g' {} \;

# Fix sleep imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.sleep import/from brain_go_brrr.domain.sleep import/g' {} \;

# Fix abnormal imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.abnormal import/from brain_go_brrr.domain.abnormal import/g' {} \;

# Fix pipeline imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.pipeline import/from brain_go_brrr.application.pipeline import/g' {} \;

# Fix jobs imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.core\.jobs/from brain_go_brrr.application.jobs/g' {} \;

# Fix tasks imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.tasks/from brain_go_brrr.application.use_cases.tasks/g' {} \;

# Fix training imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.training/from brain_go_brrr.application.training/g' {} \;

# Fix services imports
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.services\.hierarchical_pipeline/from brain_go_brrr.application.pipeline.hierarchical_pipeline/g' {} \;
find src -name "*.py" -not -path "src/brain_go_brrr/core/*" -exec sed -i 's/from brain_go_brrr\.services\.yasa_adapter/from brain_go_brrr.infra.external.yasa_adapter/g' {} \;

echo "Import fixes complete!"
