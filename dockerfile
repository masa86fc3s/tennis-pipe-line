# ✅ 正しいイメージに変更
FROM 763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.13.1-cpu-py39

RUN pip install --upgrade pip && \
    pip install \
        pandas \
        numpy \
        matplotlib \
        seaborn \
        scikit-learn \
        lightgbm==4.3.0 \
        optuna \
        boto3 \
        mypy \
        flake8 \
        sagemaker==2.230.0 \
        PyYAML


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR /opt/ml/code
