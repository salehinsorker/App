import streamlit as st
from openai import OpenAI

# Streamlit UI Configuration
st.set_page_config(page_title="Groq Chat App", page_icon="⚡", layout="centered")
st.title("⚡ Groq API (OpenAI Client) App")

# API Key Input
api_key = st.sidebar.text_input("Groq API Key (gsk-...):", type="password")

if st.button("মেসেজ পাঠান"):
    if not api_key:
        st.error("❌ দয়া করে সাইডবারে আপনার Groq API Key প্রদান করুন।")
    else:
        try:
            # Configuring OpenAI Client for Groq Endpoints
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://api.groq.com/openai/v1"
            )

            with st.spinner("উত্তর তৈরি করা হচ্ছে..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": "বাংলায় আমাকে হ্যালো বলো আর এক লাইন উৎসাহ দাও!"
                        }
                    ]
                )

            # Display Output
            output_text = response.choices[0].message.content
            st.success("🤖 AI-এর উত্তর:")
            st.write(output_text)

        except Exception as e:
            st.error(f"❌ সমস্যা হয়েছে: {str(e)}")
            
