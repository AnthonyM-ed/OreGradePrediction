import React, { useState } from "react";
import {
  TableChart,
  AccountCircle,
  Menu as MenuIcon,
  Close as CloseIcon,
  ExpandLess,
  ExpandMore
} from "@mui/icons-material";
import LogoutIcon from "@mui/icons-material/Logout";
import FolderOpen from "@mui/icons-material/FolderOpen";
import Category from "@mui/icons-material/Category";
import "./sidebar.css";

const CustomSidebar = ({ sidebarStructure, onTableSelect, collapsed, onCollapseToggle, selectedTable }) => {
  const [openSections, setOpenSections] = useState({});
  const [activeTable, setActiveTable] = useState(selectedTable || null);

  const toggleCollapse = () => {
    onCollapseToggle();
  };

  const toggleSection = (key) => {
    setOpenSections((prev) => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  return (
    <div className={`custom-sidebar ${collapsed ? "collapsed" : ""}`}>
      {/* Logo */}
      <div className="logo">
        <img
          src="/MB_logos.png"
          alt="Logo"
          className={collapsed ? "collapsed-logo" : ""}
        />
      </div>

      {/* Botón de colapsar */}
      <div className="collapse-button-wrapper">
        <button onClick={toggleCollapse} className="button-collapsed">
          {collapsed ? <MenuIcon /> : <CloseIcon />}
        </button>
      </div>

      {/* Menú jerárquico */}
      <div className="menu-wrapper">
        <nav className="menu">
          {sidebarStructure.length === 0 ? (
            <div className="no-tables-message">
              <p>No hay tablas disponibles</p>
              {/* Puedes reemplazar esto con una imagen si prefieres */}
              <img src="/path/to/no-tables-image.png" alt="No Tables" />
            </div>
          ) : (
            sidebarStructure.map((section, i) => {
              const sectionKey = `section-${i}`;
              const isOpen = openSections[sectionKey];

              return (
                <div key={sectionKey}>
                  <button
                    className="sidebar-group-title"
                    onClick={() => toggleSection(sectionKey)}
                    disabled={collapsed}
                  >
                    <div className="flex items-center">
                      <FolderOpen />
                      {!collapsed && (
                        <span style={{ marginLeft: "8px" }}>{section.title}</span>
                      )}
                    </div>
                    {!collapsed && (isOpen ? <ExpandLess /> : <ExpandMore />)}
                  </button>

                  {!collapsed && isOpen && (
                    <ul className="sidebar-subgroup">
                      {section.subItems.map((subItem, j) => {
                        const subKey = `${sectionKey}-sub-${j}`;
                        const isSubOpen = openSections[subKey];

                        return (
                          <li key={subKey}>
                            <button
                              className="sidebar-subitem"
                              onClick={() => toggleSection(subKey)}
                            >
                              <div className="flex items-center">
                                <Category style={{ marginRight: "6px", fontSize: 18 }} />
                                {subItem.title}
                              </div>
                              {subItem.tables.length > 0 && (
                                isSubOpen ? <ExpandLess /> : <ExpandMore />
                              )}
                            </button>

                            {isSubOpen && subItem.tables.length > 0 && (
                              <ul className={`table-list ${isSubOpen ? "open" : "closed"}`}>
                                {subItem.tables.map((table, t) => (
                                  <li key={t}>
                                    <button
                                      className={`sidebar-subtables ${activeTable === table ? "active" : ""}`}
                                      onClick={() => {
                                        onTableSelect(table);
                                        setActiveTable(table);
                                      }}
                                    >
                                      <div className="flex items-center">
                                        <TableChart style={{ marginRight: "6px", fontSize: 18 }} />
                                        {table}
                                      </div>
                                    </button>
                                  </li>
                                ))}
                              </ul>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              );
            })
          )}

          {/* Item Usuario */}
          <button
            className={`flex items-center user-button ${collapsed ? 'justify-center' : 'justify-between'}`}
          >
            <div className="flex items-center">
              <AccountCircle />
              {!collapsed && <span style={{ marginLeft: "8px" }}>Usuario</span>}
            </div>
            {!collapsed && <LogoutIcon />}
          </button>
        </nav>
      </div>
    </div>
  );
};

export default CustomSidebar;
