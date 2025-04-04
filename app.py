import streamlit as st
import streamlit.components.v1 as components


st.image("mainpic.jpg", width=1000)


st.markdown("""
    <div style='text-align: center;'>
        <p style='font-size: 70px; color: purple;'>🔮РАСКЛАД ГОТОВ🔮</p>
    </div>
""", unsafe_allow_html=True
)

if st.button("УЗНАТЬ СУДЬБУ"):

    st.image("tarot.gif", width=1000)

    st.markdown("""
        <div style='text-align: center;'>
            <p style='font-size: 50px; color: red;'>☢️СРОЧНАЯ ЭВАКУАЦИЯ☢️</p>
        </div>
    """, unsafe_allow_html=True
    )

    st.markdown("""
        <div style='text-align: center;'>
            <p style='font-size: 50px; color: red;'>☠️ВАМ ВЫПАЛ СТАЛАКТИТ☠️</p>
        </div>
    """, unsafe_allow_html=True
    )


    st.markdown("""
        <div style='text-align: center;'>
            <p style='font-size: 30px; color: white;'>🤕найди пострадавших во вкладке people🤕</p>
        </div>
    """, unsafe_allow_html=True
    )

    st.markdown("""
        <div style='text-align: center;'>
            <p style='font-size: 30px; color: white;'>🌳ищи укрытие на вкладке forest🌳</p>
        </div>
    """, unsafe_allow_html=True
    )
    st.markdown("""
        <div style='text-align: center;'>
            <p style='font-size: 30px; color: white;'>⚓узнай куда плыть на вкладке boats⚓</p>
        </div>
    """, unsafe_allow_html=True
    )

    st.markdown("""
        <div style='text-align: center;'>
            <p style='font-size: 130px; color: white;'>↖️</p>
        </div>
    """, unsafe_allow_html=True
    )


 