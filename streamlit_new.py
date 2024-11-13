import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from transformers import AutoConfig
import torch.nn as nn
import torch.nn.functional as F
import os

# 장치 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# LLMEmbed 모델 정의
class DownstreamModel(nn.Module):
    def __init__(self, class_num, SIGMA):
        super(DownstreamModel, self).__init__()
        self.SIGMA = SIGMA
        self.compress_layers = nn.ModuleList()
        for _ in range(5):
            layers = []
            layers.append(nn.Linear(4096, 1024))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            self.compress_layers.append(nn.Sequential(*layers))

        self.fc1 = nn.Linear(4145, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_l, input_b, input_r):
        batch_size = input_l.shape[0]

        split_tensors = torch.split(input_l, 1, dim=1)
        input = []
        for i, split_tensor in enumerate(split_tensors):
            split_tensor = split_tensor.reshape(batch_size, -1)
            input.append(self.compress_layers[i](split_tensor))

        input.append(input_b)
        input.append(input_r)
        input = torch.stack(input, dim=1)
        input_T = input.transpose(1, 2)
        input_P = torch.matmul(input, input_T)
        input_P = input_P.reshape(batch_size, -1)
        input_P = 2 * torch.sigmoid(self.SIGMA * input_P) - 1

        a = torch.mean(input_l, dim=1)
        input = torch.cat([input_P, a], dim=1)

        output = self.fc1(input)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc3(output)
        output = self.softmax(output)
        return output

# 모델 로드 함수 (Hugging Face와 로컬 모델을 지원)
@st.cache_resource
def load_classifier(model_path, is_local=False, model_type='default'):
    try:
        if model_type == 'llmembed':
            # LLMEmbed 모델 로드
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')


            llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token="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", trust_remote_code=True)
            llama2_tokenizer.pad_token = llama2_tokenizer.eos_token  # 패딩 토큰 설정
            llama2_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B",use_auth_token="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", output_hidden_states=True)
            llama2_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",use_auth_token="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", config=llama2_config)

            # llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=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, trust_remote_code=True)
            # llama2_tokenizer.pad_token = llama2_tokenizer.eos_token  # 패딩 토큰 설정
            # llama2_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=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).to(device)
            bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")
            bert_model = BertModel.from_pretrained("google-bert/bert-large-uncased").to(device)
            roberta_tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")
            roberta_model = RobertaModel.from_pretrained("FacebookAI/roberta-large").to(device)
            downstream_model = DownstreamModel(class_num=5, SIGMA=0.1).to(device)

            model_load_path = "/Users/suwon/Desktop/ML/ai-algorithm/model_weights_stackexchange.pth"
            downstream_model.load_state_dict(torch.load(model_load_path, map_location=device))
            downstream_model.eval()

            st.success(f"LLMEmbed 모델 로드 성공")
            return {
                'llama2_tokenizer': llama2_tokenizer,
                'llama2_model': llama2_model,
                'bert_tokenizer': bert_tokenizer,
                'bert_model': bert_model,
                'roberta_tokenizer': roberta_tokenizer,
                'roberta_model': roberta_model,
                'downstream_model': downstream_model,
                'device': device
            }
        else:
            if is_local:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
                st.success(f"로컬 모델 로드 성공")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                st.success(f"모델 로드 성공")
            return {'tokenizer': tokenizer, 'model': model}
    except Exception as e:
        st.error(f"모델 또는 토크나이저 로딩 중 오류 발생: {e}")

def classify_text(text, tokenizer, model, labels, multi_label=False):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    if multi_label:
        scores = torch.sigmoid(logits).squeeze().tolist()
    else:
        scores = torch.softmax(logits, dim=1).squeeze().tolist()
    result = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    return result

# LLMEmbed 모델 추론 함수
def infer_llmembed(text, model_components):
    device = model_components['device']
    llama2_tokenizer = model_components['llama2_tokenizer']
    llama2_model = model_components['llama2_model']
    bert_tokenizer = model_components['bert_tokenizer']
    bert_model = model_components['bert_model']
    roberta_tokenizer = model_components['roberta_tokenizer']
    roberta_model = model_components['roberta_model']
    downstream_model = model_components['downstream_model']

    # 각 모델로부터 임베딩을 추출
    llama2_emb = get_llama2_embedding(text, llama2_tokenizer, llama2_model, device)
    bert_emb = get_bert_embedding(text, bert_tokenizer, bert_model, device)
    roberta_emb = get_roberta_embedding(text, roberta_tokenizer, roberta_model, device)

    # Forward pass through the downstream model
    with torch.no_grad():
        prediction = downstream_model(llama2_emb, bert_emb, roberta_emb)
        scores = prediction.squeeze().tolist()
        result = sorted(zip(label_map.values(), scores), key=lambda x: x[1], reverse=True)
    return result

def get_llama2_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 마지막 5개 레이어의 평균을 사용
        hidden_states = outputs.hidden_states[-5:]
        embedding = torch.mean(torch.stack(hidden_states), dim=0).mean(dim=1)
    return embedding

def get_bert_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output
    return embedding

def get_roberta_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

import streamlit as st

def main():

    st.markdown("""
        <style>
            .title {
                font-size: 2.5em;
                color: #2A75AF;
                font-weight: bold;
                text-align: center;
                margin-bottom: 0.2em;
            }
            .subtitle {
                font-size: 1.2em;
                color: #444;
                text-align: center;
                margin-top: 0;
                margin-bottom: 1.5em;
            }
            .description {
                background-color: #e6f7f9;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                font-size: 1em;
                color: #333;
                margin-bottom: 1.5em;
            }
        </style>
        
        <div class="title">SYNexis: Text Classification Platform</div>
        <div class="subtitle">A Collaborative Innovation by Samsung SDS and Yonsei University</div>

        <div class="description">
            <p><b>SYNexis</b> harnesses advanced machine learning to classify text inputs with accuracy and insight, delivering smart solutions for complex data interpretation.</p>
            <p>Developed through a unique partnership between <b>Samsung SDS</b> and <b>Yonsei University</b>, SYNexis stands at the forefront of intelligent systems.</p>
            <p>Explore the power of AI and experience the future of data classification.</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("Enter your message below to receive customized AI-driven classification results:")

    # 설정 확장 탭
    with st.expander("🔧 Settings"):
        st.write("Choose models and start classification.")
        
        # 모델 선택 옵션 정의
        model_options = {
            "Multi Class Sobert": ("/Users/suwon/Desktop/ML/ai-algorithm/single_cls_sobert_base", True, 'default'),
            "Multi Label Sobert": ("/Users/suwon/Desktop/ML/ai-algorithm/multi_cls_sobert_base", True, 'default'),
            "Hugging Face 모델 (ebinna/multi_cls_mamba2-130m)": ("ebinna/multi_cls_mamba2-130m", False, 'default'),
            "LLMEmbed 모델": ("llmembed", True, 'llmembed')
        }

        # 모델 선택 탭 구성
        tabs = st.tabs(["Multi-Class Model Selection", "Multi-Label Model Selection"])
        with tabs[0]:
            st.write("**Multi-Class Model**")
            multi_class_choice = st.selectbox("Choose a Multi-Class Classification Model:", list(model_options.keys()), key="multi_class_model")
            multi_class_path, multi_class_is_local, multi_class_type = model_options[multi_class_choice]
            multi_class_model_components = load_classifier(multi_class_path, is_local=multi_class_is_local, model_type=multi_class_type)

        with tabs[1]:
            st.write("**Multi-Label Model**")
            multi_label_choice = st.selectbox("Choose a Multi-Label Classification Model:", list(model_options.keys()), key="multi_label_model")
            multi_label_path, multi_label_is_local, multi_label_type = model_options[multi_label_choice]
            multi_label_model_components = load_classifier(multi_label_path, is_local=multi_label_is_local, model_type=multi_label_type)

    post = st.text_area("✍️ Compose your text here:", placeholder="Type your message here and press Submit.")

    if st.button("Submit"):
        if post:
            with st.spinner("Classifying..."):
                # 세션 상태 초기화
                st.session_state.primary_label = None
                st.session_state.primary_score = None
                st.session_state.multi_label_labels = []
                st.session_state.second_result = []

                # Multi-Class Classification 수행
                if multi_class_choice == "LLMEmbed 모델":
                    result = infer_llmembed(post, multi_class_model_components)
                else:
                    tokenizer = multi_class_model_components['tokenizer']
                    model = multi_class_model_components['model']
                    result = classify_text(post, tokenizer, model, first_labels)
                primary_label = result[0][0]

                # 세션 상태 저장
                st.session_state.primary_label = primary_label
                st.session_state.primary_score = result[0][1]

                if primary_label.lower() == "stackoverflow":
                    # Multi-Label Classification 수행
                    if multi_label_choice == "LLMEmbed 모델":
                        second_result = infer_llmembed(post, multi_label_model_components)
                    else:
                        tokenizer = multi_label_model_components['tokenizer']
                        model = multi_label_model_components['model']
                        second_result = classify_text(post, tokenizer, model, second_labels, multi_label=True)
                    multi_label_labels = [label for label, score in second_result if score >= 0.5]

                    st.session_state.multi_label_labels = multi_label_labels
                    st.session_state.second_result = second_result

    # 결과 출력
    st.write("---")
    if "primary_label" in st.session_state and st.session_state.primary_label:
        st.subheader("📌 Classification Results")
        st.markdown(f"**1. Multi-Class Classification:** :blue[{st.session_state.primary_label.capitalize()}]")

    if "multi_label_labels" in st.session_state and st.session_state.multi_label_labels:
        st.markdown("**2. Multi-Label Classification:**")
        st.write(", ".join([f":green[{label}]" for label in st.session_state.multi_label_labels]))

    # Score 확인 옵션 제공
    if "primary_label" in st.session_state and st.session_state.primary_label and st.checkbox("Show Scores"):
        st.subheader("📊 Score")
        st.markdown("**Multi-Class Classification Score:**")
        st.write(f"{st.session_state.primary_label.capitalize()}: {st.session_state.primary_score:.4f}")

        st.markdown("**Multi-Label Classification Scores:**")
        for label, score in st.session_state.second_result:
            st.write(f"{label}: {score:.4f}")

# 레이블 맵
label_map = {
    0: "biology",
    1: "cooking",
    2: "diy",
    3: "travel",
    4: "stackoverflow"
}

if __name__ == '__main__':
    # 분류 레이블 설정
    first_labels = ['biology', 'cooking', 'diy', 'travel', 'stackoverflow']
    second_labels = ['Algorithms', 'Backend', 'Data Science', 'Databases', 'Dev Tools', 'Frontend', 'Mobile', 'Systems', 'iOS/macOS']
    main()
