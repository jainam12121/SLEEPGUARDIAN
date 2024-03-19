import pytest
from flask import session
from server import app # Assuming 'server' is the name of your Flask application file

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_session_handling(client):
    with client.session_transaction() as session:
        session['username'] = 'testuser'
    response = client.post('/login')
    assert session.get('username') == 'testuser'
