// Authentication logic
document.addEventListener('DOMContentLoaded', function () {
    const loginForm = document.getElementById('login-form');

    // Handle login form submission
    if (loginForm) {
        loginForm.addEventListener('submit', function (e) {
            e.preventDefault();

            // Get form data
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;

            // Hardcoded credentials for testing
            const validUsername = "admin";
            const validPassword = "password";

            // Simple validation
            if (username === validUsername && password === validPassword) {
                // Save login state in session storage
                sessionStorage.setItem('isLoggedIn', 'true');

                // Redirect to home page
                window.location.href = 'home.html';
            } else {
                alert('Invalid username or password. Please try again.');
            }
        });
    }

    // Check if user is already logged in
    if (sessionStorage.getItem('isLoggedIn') === 'true') {
        // Redirect to home.html if user is logged in and tries to access login.html
        if (window.location.pathname.endsWith('login.html')) {
            window.location.href = 'home.html';
        }
    } else {
        // Redirect to login.html if user is not logged in and tries to access protected pages
        const protectedPages = ['index.html', 'home.html', 'appointments.html', 'logout.html'];
        if (protectedPages.some(page => window.location.pathname.endsWith(page))) {
            window.location.href = 'login.html';
        }
    }
});