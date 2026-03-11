pipeline {
    options {
        timestamps()
        skipDefaultCheckout()
        disableConcurrentBuilds()
    }
    agent {
        node { label 'translator && build && aws' }
    }
    parameters {
        string(name: 'BUILD_VERSION', defaultValue: '', description: 'The build version to deploy (optional)')
        string(name: 'AWS_REGION', defaultValue: 'us-east-1', description: 'AWS Region to deploy')
    }
    triggers {
        pollSCM('H/5 * * * *')
    }
    environment {
        IMAGE_NAME = "853771734544.dkr.ecr.us-east-1.amazonaws.com/retriever"
        KUBERNETES_BLUE_CLUSTER_NAME = "translator-eks-ci-blue-cluster"
        DEPLOY_ENV = "ci"
        NAMESPACE = 'retriever'
    }
    stages {
        stage('Build Version'){
            when { expression { return !params.BUILD_VERSION } }
            steps{
                script {
                    BUILD_VERSION_GENERATED = VersionNumber(
                        versionNumberString: 'v${BUILD_YEAR, XX}.${BUILD_MONTH, XX}${BUILD_DAY, XX}.${BUILDS_TODAY}',
                        projectStartDate:    '1970-01-01',
                        skipFailedBuilds:    true)
                    currentBuild.displayName = BUILD_VERSION_GENERATED
                    env.BUILD_VERSION = BUILD_VERSION_GENERATED
                    env.BUILD = 'true'
                }
            }
        }
        stage('Checkout source code') {
            steps {
                cleanWs()
                checkout scm
            }
        }
        stage('Build Docker') {
           when { expression { return env.BUILD == 'true' }}
            steps {
                script {
                    sh '''#!/bin/bash
                    echo "Building Docker image for retriever..."
                    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin 853771734544.dkr.ecr.us-east-1.amazonaws.com
                    docker build -t retriever:${BUILD_VERSION} .
                    docker tag retriever:${BUILD_VERSION} ${IMAGE_NAME}:${BUILD_VERSION}
                    docker tag retriever:${BUILD_VERSION} ${IMAGE_NAME}:latest
                    docker push ${IMAGE_NAME}:${BUILD_VERSION}
                    docker push ${IMAGE_NAME}:latest
                    '''
                }
            }
        }
        stage('Deploy to AWS EKS Blue') {
            agent {
                label 'translator && ci && deploy'
            }
            steps {
                script {
                    configFileProvider([
                        configFile(fileId: 'values-retriever-ci.yaml', targetLocation: 'values-ncats.yaml')
                    ]){
                        sh '''
                        aws --region ${AWS_REGION} eks update-kubeconfig --name ${KUBERNETES_BLUE_CLUSTER_NAME}
                        cd translator-ops/ops/retriever/
                        /bin/bash deploy.sh
                        '''
                    }
                }
            }
            post {
                always {
                    echo "Clean up the workspace in deploy node!"
                    cleanWs()
                }
            }
        }
    }
    post {
        success {
            echo "Pipeline completed successfully!"
            echo "Image: ${IMAGE_NAME}:${BUILD_VERSION}"
        }
        failure {
            echo "Pipeline failed!"
        }
    }
}
