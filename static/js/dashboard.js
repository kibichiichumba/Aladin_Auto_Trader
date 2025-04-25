document.addEventListener('DOMContentLoaded', function() {
    // Notification polling
    let notificationBadge = document.querySelector('.notification-badge');
    let notificationsContainer = document.querySelector('.notifications-container');
    let notificationDropdown = document.querySelector('.notification-dropdown');
    
    // Function to format datetime
    function formatDateTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString();
    }
    
    // Function to update notifications
    function updateNotifications() {
        fetch('/api/notifications')
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    // Update badge
                    notificationBadge.textContent = data.length;
                    notificationBadge.classList.remove('d-none');
                    
                    // Clear existing notifications
                    document.querySelector('.no-notifications')?.remove();
                    
                    // Clear and update notifications container
                    notificationsContainer.innerHTML = '';
                    
                    // Add notifications
                    data.forEach(notification => {
                        const notificationItem = document.createElement('li');
                        
                        // Set notification type class
                        let typeClass = 'bg-primary';
                        if (notification.type === 'trade') typeClass = 'bg-success';
                        if (notification.type === 'error') typeClass = 'bg-danger';
                        if (notification.type === 'risk') typeClass = 'bg-warning';
                        
                        notificationItem.innerHTML = `
                            <div class="dropdown-item">
                                <div class="d-flex align-items-center">
                                    <div class="flex-shrink-0">
                                        <div class="rounded-circle p-1 ${typeClass}">
                                            <i class="fas fa-${notification.type === 'trade' ? 'chart-line' : notification.type === 'error' ? 'exclamation-circle' : notification.type === 'risk' ? 'shield-alt' : 'bell'} text-white"></i>
                                        </div>
                                    </div>
                                    <div class="ms-3">
                                        <div class="fw-bold">${notification.title}</div>
                                        <div class="small text-muted">${notification.message}</div>
                                        <div class="small text-muted">${formatDateTime(notification.timestamp)}</div>
                                    </div>
                                    <div class="ms-2">
                                        <button class="btn btn-sm btn-link text-muted mark-read" data-id="${notification.id}">
                                            <i class="fas fa-times"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        `;
                        
                        notificationsContainer.appendChild(notificationItem);
                    });
                    
                    // Add event listeners to mark read buttons
                    document.querySelectorAll('.mark-read').forEach(button => {
                        button.addEventListener('click', function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            
                            const notificationId = this.getAttribute('data-id');
                            markNotificationRead(notificationId);
                        });
                    });
                } else {
                    // No notifications
                    notificationBadge.classList.add('d-none');
                    notificationsContainer.innerHTML = '<li class="dropdown-item text-center text-muted no-notifications">No new notifications</li>';
                }
            })
            .catch(error => {
                console.error('Error fetching notifications:', error);
            });
    }
    
    // Function to mark notification as read
    function markNotificationRead(notificationId) {
        fetch(`/api/notifications/read/${notificationId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update notifications
                updateNotifications();
            }
        })
        .catch(error => {
            console.error('Error marking notification as read:', error);
        });
    }
    
    // Mark all notifications as read
    document.getElementById('markAllRead')?.addEventListener('click', function(e) {
        e.preventDefault();
        
        document.querySelectorAll('.mark-read').forEach(button => {
            const notificationId = button.getAttribute('data-id');
            markNotificationRead(notificationId);
        });
    });
    
    // Initial notification check
    if (notificationBadge) {
        updateNotifications();
        
        // Poll for new notifications every 30 seconds
        setInterval(updateNotifications, 30000);
    }
    
    // Candlestick Chart Extension for Chart.js
    if (typeof Chart !== 'undefined') {
        // This is handled by chart_utils.js, so we can safely remove it here
        // to avoid conflicts with the implementation in chart_utils.js
    }
});
