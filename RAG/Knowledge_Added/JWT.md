## JWT（JSON Web Token）

JWT是一种**开放标准（RFC 7519）**，用于在网络应用间安全传输声明信息。它本质上是一个**数字令牌**，包含JSON格式的数据，通常用于身份验证和授权。

**JWT的结构（三部分用`.`连接）：**

```text
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

- **Header（头部）**：包含令牌类型（JWT）和签名算法（如HS256）

```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

- **Payload（载荷）**：包含声明（claims），即需要传递的实际数据

```json
{
  "sub": "1234567890",  // 主题（用户ID）
  "name": "John Doe",   // 自定义声明
  "iat": 1516239022     // 签发时间
}
```

- **Signature（签名）**：对头部和载荷的签名，防止数据被篡改

```javascript
HMACSHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  secret_key
)
```

## JWT VS Session机制

首先需要理解HTTP协议是无状态的，这意味着每次请求都是独立的，服务器不会记住之前的请求。但是，很多应用需要保持用户登录状态，这就需要一种机制来维持状态。Session和JWT是两种不同的解决方案。

### Session机制

**工作流程：**

1. 用户登录，服务器验证用户名和密码。
2. 服务器创建一个Session，将用户信息存储在服务器内存或数据库（如Redis）中，并生成一个唯一的Session ID。
3. 服务器将Session ID通过Cookie返回给客户端（设置Set-Cookie头）。
4. 客户端后续请求自动携带这个Cookie（浏览器自动处理）。
5. 服务器通过Session ID查找对应的Session，获取用户信息。

**特点：**

- **有状态**：服务器需要存储Session数据。
- **默认使用Cookie传输**，但也可以使用URL重写（但不安全）。
- 容易受到CSRF（跨站请求伪造）攻击，因为浏览器会自动发送Cookie。

### JWT（JSON Web Token）机制

**工作流程：**

1. 用户登录，服务器验证用户名和密码。
2. 服务器生成一个JWT，将用户信息（claims）和过期时间等放入Payload，并用密钥签名。
3. 服务器将JWT返回给客户端，客户端保存（通常放在localStorage或Cookie中）。
4. 客户端后续请求手动携带JWT（通常在Authorization头中）。
5. 服务器验证JWT的签名和有效期，从Payload中直接读取用户信息。

**特点：**

- **无状态**：服务器不需要存储Token，因为所有信息都在Token中，且签名保证不被篡改。
- 客户端需要主动将Token放入请求头（或Cookie，但不会自动发送，可避免CSRF）。
- Token一旦签发，在过期前一直有效（除非服务器额外维护黑名单）。

### CRSF(跨站请求伪造)

CSRF（Cross-Site Request Forgery，跨站请求伪造）是一种攻击方式，攻击者诱导受害者在已登录目标网站的情况下，访问恶意网站或点击恶意链接，从而以受害者的身份执行非本意的操作（如转账、改密码等）。

攻击原理：

1. 用户登录了信任的网站A，并在浏览器中保存了登录凭证（如Session Cookie）。
2. 用户在没有登出网站A的情况下，访问了恶意网站B。
3. 网站B中可能包含一个指向网站A的请求（例如，一个图片标签，其src是网站A的转账接口），当用户访问B时，浏览器会自动携带网站A的Cookie，从而以用户的身份执行操作。

防护措施：

- **使用CSRF Token**：服务器生成一个随机Token，放在表单或请求头中，每次请求验证此Token。
- **SameSite Cookie属性**：设置Cookie的SameSite属性为Strict或Lax，限制跨站请求携带Cookie。
- **验证HTTP Referer/Origin头**：检查请求来源是否合法。
- **对于敏感操作使用二次确认**（如短信验证码、密码确认）。