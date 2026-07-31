import { useState } from 'react';
import { Routes, Route, Navigate, useLocation, Link } from 'react-router-dom';
import { Layout, Menu, theme as antdTheme } from 'antd';
import {
  SettingOutlined,
  CloudUploadOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  MessageOutlined,
  LikeOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import ConfigPage from './pages/ConfigPage';
import IngestPage from './pages/IngestPage';
import DocumentsPage from './pages/DocumentsPage';
import KnowledgeBasePage from './pages/KnowledgeBasePage';
import QaPage from './pages/QaPage';
import FeedbackPage from './pages/FeedbackPage';

const { Header, Sider, Content } = Layout;

const menuItems: MenuProps['items'] = [
  { key: '/config', icon: <SettingOutlined />, label: <Link to="/config">配置管理</Link> },
  { key: '/ingest', icon: <CloudUploadOutlined />, label: <Link to="/ingest">文档上传</Link> },
  { key: '/documents', icon: <FileTextOutlined />, label: <Link to="/documents">文档管理</Link> },
  { key: '/knowledge-base', icon: <DatabaseOutlined />, label: <Link to="/knowledge-base">知识库浏览</Link> },
  { key: '/qa', icon: <MessageOutlined />, label: <Link to="/qa">问答测试</Link> },
  { key: '/feedback', icon: <LikeOutlined />, label: <Link to="/feedback">反馈统计</Link> },
];

function selectedKey(pathname: string): string {
  const match = menuItems?.find((it) => pathname.startsWith(String((it as { key: string }).key)));
  return match ? (match as { key: string }).key : '/config';
}

export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  const { token } = antdTheme.useToken();

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={200}
        collapsedWidth={80}
        theme="light"
        style={{ borderRight: `1px solid ${token.colorBorderSecondary}` }}
      >
        <div
          style={{
            height: 56,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 700,
            color: token.colorPrimary,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
          }}
        >
          {collapsed ? 'SS' : 'SparkSage'}
        </div>
        <Menu
          mode="vertical"
          selectedKeys={[selectedKey(location.pathname)]}
          items={menuItems}
          style={{ borderInlineEnd: 'none' }}
        />
      </Sider>
      <Layout>
        <Header
          style={{
            background: token.colorBgContainer,
            padding: '0 16px',
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            borderBottom: `1px solid ${token.colorBorderSecondary}`,
          }}
        >
          <span
            onClick={() => setCollapsed((c) => !c)}
            style={{ cursor: 'pointer', fontSize: 18 }}
          >
            {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          </span>
          <strong>SparkSage 演示工具</strong>
        </Header>
        <Content style={{ padding: 24, background: token.colorBgLayout }}>
          <Routes>
            <Route path="/" element={<Navigate to="/config" replace />} />
            <Route path="/config" element={<ConfigPage />} />
            <Route path="/ingest" element={<IngestPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/knowledge-base" element={<KnowledgeBasePage />} />
            <Route path="/qa" element={<QaPage />} />
            <Route path="/feedback" element={<FeedbackPage />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}
