import streamlit as st
from typing import Dict

def render_user_profile(profile: Dict):
    """Renders the user profile panel."""
    st.markdown("""
        <div style="background: #0f1628; border: 1px solid rgba(179, 197, 255, 0.14); border-radius: 16px; padding: 20px; margin-bottom: 30px;">
            <h3 style="margin-top: 0; color: #b3c5ff; font-weight: 800; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                👤 Reader Profile
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(179,197,255,0.08);">
                    <div style="color: #9aa6c4; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">Favorite Genres</div>
                    <div style="color: #eef3ff; font-weight: bold; margin-top: 4px;">{}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(179,197,255,0.08);">
                    <div style="color: #9aa6c4; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">Favorite Authors</div>
                    <div style="color: #eef3ff; font-weight: bold; margin-top: 4px;">{}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(179,197,255,0.08);">
                    <div style="color: #9aa6c4; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">Books Rated</div>
                    <div style="color: #00f1fe; font-weight: bold; font-size: 1.2rem; margin-top: 4px;">{}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(179,197,255,0.08);">
                    <div style="color: #9aa6c4; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">Avg Rating Given</div>
                    <div style="color: #ffd700; font-weight: bold; font-size: 1.2rem; margin-top: 4px;">{:.1f} ★</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(179,197,255,0.08);">
                    <div style="color: #9aa6c4; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">Reading Diversity</div>
                    <div style="color: #b3c5ff; font-weight: bold; font-size: 1.2rem; margin-top: 4px;">{} Genres</div>
                </div>
            </div>
        </div>
    """.format(
        ", ".join(profile.get("Favorite_Genres", []))[:60] + "..." if profile.get("Favorite_Genres") else "None",
        ", ".join(profile.get("Favorite_Authors", []))[:60] + "..." if profile.get("Favorite_Authors") else "None",
        profile.get("Books_Rated", 0),
        profile.get("Average_Rating", 0.0),
        profile.get("Reading_Diversity", 0)
    ), unsafe_allow_html=True)
