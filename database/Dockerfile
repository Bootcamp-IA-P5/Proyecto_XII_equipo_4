# Database Dockerfile for Logo Detection MySQL
FROM mysql:8.0

# Set working directory
WORKDIR /docker-entrypoint-initdb.d

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/01-init.sql

# Set environment variables
ENV MYSQL_DATABASE=logo_detection
ENV MYSQL_USER=logo_user
ENV MYSQL_PASSWORD=logo_password
ENV MYSQL_ROOT_PASSWORD=root_password

# MySQL configuration for better performance
RUN echo "[mysqld]" >> /etc/mysql/conf.d/custom.cnf && \
    echo "max_connections=200" >> /etc/mysql/conf.d/custom.cnf && \
    echo "innodb_buffer_pool_size=256M" >> /etc/mysql/conf.d/custom.cnf && \
    echo "innodb_log_file_size=64M" >> /etc/mysql/conf.d/custom.cnf && \
    echo "character-set-server=utf8mb4" >> /etc/mysql/conf.d/custom.cnf && \
    echo "collation-server=utf8mb4_unicode_ci" >> /etc/mysql/conf.d/custom.cnf

# Expose MySQL port
EXPOSE 3306

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD mysqladmin ping -h localhost -u root -p${MYSQL_ROOT_PASSWORD} || exit 1
