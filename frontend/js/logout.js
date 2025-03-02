// Logout functionality
document.addEventListener('DOMContentLoaded', function () {
    const confirmButton = document.querySelector('.logout-button.confirm');
    const cancelButton = document.querySelector('.logout-button.cancel');

    // Check if the logout page is loaded
    if (window.location.pathname.endsWith('logout.html')) {
        // Handle Logout Confirmation
        if (confirmButton) {
            confirmButton.addEventListener('click', function () {
                // Clear login state
                sessionStorage.removeItem('isLoggedIn');

                // Show a confirmation message
                alert('You have been logged out successfully.');

                // Redirect to login page
                window.location.href = 'login.html';
            });
        }

        // Handle Cancel
        if (cancelButton) {
            cancelButton.addEventListener('click', function () {
                // Redirect back to home page
                window.location.href = 'home.html';
            });
        }

        // Check if user is not logged in
        if (sessionStorage.getItem('isLoggedIn') !== 'true') {
            // Redirect to login page if not logged in
            window.location.href = 'login.html';
        }
    }
});